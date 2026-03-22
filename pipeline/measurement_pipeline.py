import os

import cv2

from utils.image_utils import validate_dual_images
from utils.debug_visualization import (
	draw_bbox,
	draw_pose,
	draw_torso_lines,
	overlay_mask,
	plot_width_profile,
	save_segmentation_with_bbox,
	show_mask,
)
from measurement.depth_estimation import compute_torso_circumferences, compute_torso_depths
from measurement.feature_scaling import (
	build_feature_vector as build_ml_feature_vector,
)
from measurement.height_measurement import compute_pixel_height
from measurement.limb_measurements import (
	compute_arm_length,
	compute_leg_length,
	compute_shoulder_to_waist,
	compute_waist_to_knee,
	distance,
)
from measurement.torso_measurements import (
	compute_torso_widths,
	compute_width_profile,
	detect_torso_measurements,
)
from vision.detection import detect_person
from vision.pose import extract_pose_keypoints
from vision.segmentation import HumanSegmenter


class MeasurementPipeline:

	def __init__(self, models):
		self.person_model = models["person_model"]
		self.pose_model = models["pose_model"]
		self.bodytype_model = models["bodytype_model"]
		self.label_encoder = models["label_encoder"]
		self.segmenter = HumanSegmenter()

	def _debug_visualization_enabled(self):
		return os.getenv("BODY_DEBUG_VIS", "0") == "1"

	def run(
		self,
		front_image,
		side_image,
		age,
		gender,
		height
	):
		"""
		Main pipeline execution.
		"""

		# 1. Validate images
		images = self.validate_images(front_image, side_image)

		# 2. Detect person
		detected = self.detect_person(images)

		# 3. Segment body
		segmented = self.segment_body(detected)

		# 4. Extract pose
		pose_data = self.extract_pose(detected)

		# 5. Compute measurements (pixel space)
		measurements_pixels = self.compute_measurements(
			segmented,
			pose_data,
		)

		# 6. Compute pixel height
		pixel_height = measurements_pixels["pixel_height"]

		# 7. Compute scaling factor
		scale = height / pixel_height

		# 8. Scale measurements
		scaled_measurements = {}

		for key, value in measurements_pixels.items():

			if key != "pixel_height":
				scaled_measurements[key] = float(value * scale)

		# 9. Compute body fat
		waist = scaled_measurements["waist"]
		body_fat = float(64 - (20 * waist / height))

		# 10. Build feature vector
		features = self.build_feature_vector(
			gender,
			age,
			scaled_measurements,
			height,
			body_fat,
		)

		# 11. Predict body type
		prediction = self.predict_body_type(features)

		return {
			"body_type": prediction["body_type"],
			"ayurvedic_type": prediction["ayurvedic_type"],
			"measurements": scaled_measurements
		}

	def validate_images(self, front_image, side_image):
		"""
		Validate images before processing.
		"""
		images = validate_dual_images(front_image, side_image)

		return images

	def detect_person(self, images):
		"""
		Run YOLO person detection.
		"""

		front_result = detect_person(
			images["front"],
			self.person_model
		)

		side_result = detect_person(
			images["side"],
			self.person_model
		)

		if self._debug_visualization_enabled():
			draw_bbox(images["front"], front_result["bbox"], filename="front_person_detection.png")
			draw_bbox(images["side"], side_result["bbox"], filename="side_person_detection.png")

		return {
			"front_crop": front_result["crop"],
			"side_crop": side_result["crop"],
			"front_bbox": front_result["bbox"],
			"side_bbox": side_result["bbox"]
		}

	def segment_body(self, images):
		"""
		Run segmentation model.
		"""

		front_mask = self.segmenter.segment(images["front_crop"])
		side_mask = self.segmenter.segment(images["side_crop"])

		return {
			"front_mask": front_mask,
			"side_mask": side_mask,
			"front_crop": images["front_crop"],
			"side_crop": images["side_crop"]
		}

	def extract_pose(self, images):
		"""
		Extract pose keypoints.
		"""

		front_pose = extract_pose_keypoints(
			images["front_crop"],
			self.pose_model
		)

		side_pose = extract_pose_keypoints(
			images["side_crop"],
			self.pose_model
		)

		return {
			"front_pose": front_pose,
			"side_pose": side_pose
		}

	def compute_measurements(self, segmentation_data, pose_data):
		"""
		Compute anthropometric measurements.
		"""
		front_mask = segmentation_data["front_mask"]
		side_mask = segmentation_data["side_mask"]
		front_crop = segmentation_data["front_crop"]
		side_crop = segmentation_data["side_crop"]
		front_pose = pose_data["front_pose"]
		side_pose = pose_data["side_pose"]
		debug_vis = self._debug_visualization_enabled()

		if debug_vis:
			show_mask(front_mask, filename="front_segmentation_mask.png")
			show_mask(side_mask, filename="side_segmentation_mask.png")
			overlay_mask(front_crop, front_mask, filename="front_mask_overlay.png")
			overlay_mask(side_crop, side_mask, filename="side_mask_overlay.png")
			save_segmentation_with_bbox(front_crop, front_mask, filename="front_segmentation_with_bbox.png")
			save_segmentation_with_bbox(side_crop, side_mask, filename="side_segmentation_with_bbox.png")
			draw_pose(front_crop, front_pose, filename="front_pose.png")
			draw_pose(side_crop, side_pose, filename="side_pose.png")

		pixel_height, _, _ = compute_pixel_height(front_mask)

		# Extract shoulder/hip pixel coordinates from pose for anatomy-guided scanning.
		left_shoulder_x = min(float(front_pose["left_shoulder"][0]), float(front_pose["right_shoulder"][0]))
		right_shoulder_x = max(float(front_pose["left_shoulder"][0]), float(front_pose["right_shoulder"][0]))
		pose_shoulder_y = float((front_pose["left_shoulder"][1] + front_pose["right_shoulder"][1]) / 2)
		pose_hip_y = float((front_pose["left_hip"][1] + front_pose["right_hip"][1]) / 2)

		width_profile = compute_width_profile(front_mask, left_shoulder_x, right_shoulder_x)
		chest_y, waist_y, hip_y = detect_torso_measurements(width_profile, pose_shoulder_y, pose_hip_y)

		if debug_vis:
			draw_torso_lines(front_crop, chest_y, waist_y, hip_y, filename="front_torso_rows.png")
			plot_width_profile(width_profile, filename="front_width_profile.png")

		torso_widths = compute_torso_widths(front_mask, left_shoulder_x, right_shoulder_x, pose_shoulder_y, pose_hip_y)
		torso_depths = compute_torso_depths(side_mask, chest_y, waist_y, hip_y)
		torso_circumferences = compute_torso_circumferences(torso_widths, torso_depths)

		shoulder_width = distance(
			front_pose["left_shoulder"],
			front_pose["right_shoulder"],
		)

		chest = torso_circumferences["chest_pixels"]
		if chest > 2 * shoulder_width:
			chest = shoulder_width * 1.5

		print("pixel height:", pixel_height)
		print("chest_y waist_y hip_y:", chest_y, waist_y, hip_y)
		print("torso_widths:", torso_widths)
		print("torso_depths:", torso_depths)
		print("circumferences:", torso_circumferences)

		measurements_pixels = {
			"pixel_height": float(pixel_height),
			"shoulder_width": shoulder_width,
			"chest": chest,
			"waist": torso_circumferences["waist_pixels"],
			"hips": torso_circumferences["hips_pixels"],
			"belly": torso_circumferences["waist_pixels"] * 1.1,
			"arm_length": compute_arm_length(front_pose),
			"shoulder_to_waist": compute_shoulder_to_waist(front_pose),
			"waist_to_knee": compute_waist_to_knee(front_pose),
			"leg_length": compute_leg_length(front_pose),
		}

		return measurements_pixels

	def build_feature_vector(self, gender, age, measurements, height, body_fat):
		"""
		Build ML feature vector.
		"""
		return build_ml_feature_vector(
			gender=gender,
			age=age,
			measurements=measurements,
			total_height=height,
			body_fat=body_fat,
		)

	def map_to_ayurvedic(self, bodytype):
		"""
		Map Western body type to Ayurvedic type.
		"""
		mapping = {
			"Ectomorph": "Vata",
			"Mesomorph": "Pitta",
			"Endomorph": "Kapha"
		}

		return mapping.get(bodytype, "Unknown")

	def predict_body_type(self, features):
		"""
		Run ML model prediction.
		"""
		prediction = self.bodytype_model.predict([features])
		label = self.label_encoder.inverse_transform(prediction)[0]

		ayurvedic_type = self.map_to_ayurvedic(label)

		return {
			"body_type": str(label),
			"ayurvedic_type": str(ayurvedic_type)
		}
