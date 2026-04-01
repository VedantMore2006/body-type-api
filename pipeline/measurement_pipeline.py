import os
import math

from utils.image_utils import validate_single_image
from utils.debug_visualization import (
	draw_bbox,
	draw_pose,
	draw_scaling_overlay,
	show_mask,
)
from measurement.feature_scaling import (
	build_feature_vector as build_ml_feature_vector,
)
from measurement.limb_measurements import (
	compute_arm_length,
	compute_leg_length,
	compute_leg_length_to_floor,
	compute_shoulder_to_waist,
	compute_waist_to_knee,
	distance,
)
from vision.detection import (
	bbox_center_x,
	bbox_height,
	bbox_width,
	detect_door_heuristic,
	detect_person,
	is_bbox_fully_visible,
)
from vision.pose import extract_pose_keypoints


class MeasurementPipeline:

	def __init__(self, models):
		self.person_model = models["person_model"]
		self.pose_model = models["pose_model"]
		self.bodytype_model = models["bodytype_model"]
		self.label_encoder = models["label_encoder"]
		self.segmenter = models["segmenter"]

	def _debug_visualization_enabled(self):
		return os.getenv("BODY_DEBUG_VIS", "0") == "1"

	def run(
		self,
		image,
		age,
		gender,
		person_height_cm=170.0,
		person_weight_kg=65.0,
	):
		"""
		Main pipeline execution for direct height-reference scaling.
		"""

		# 1. Validate input image
		payload = self.validate_image(image)
		frame = payload["image"]

		# 2. Detect person in the frame
		detection = self.detect_person_only(frame)
		person_bbox = detection["person_bbox"]

		# 3. Guardrails for common geometric failures
		self.validate_person_geometry(frame, person_bbox)

		# 4. Extract pose and mask from person crop
		person_crop = detection["person_crop"]
		pose_data = self.extract_pose(person_crop)
		front_pose = pose_data["front_pose"]
		person_mask = self.segmenter.segment(person_crop)

		if self._debug_visualization_enabled():
			show_mask(person_mask)

		# 5. Compute tilt and vertical correction
		tilt_angle = self._detect_tilt(front_pose)
		v_factor = self._get_vertical_correction_factor(tilt_angle)

		# 6. Compute scale from known person height
		# Apply vertical factor to the pixel height to get the 'straight' pixel height
		person_height_px = float(bbox_height(person_bbox)) * v_factor
		if person_height_px <= 0:
			raise ValueError("Invalid person height in pixels")

		scale_cm_per_px = float(person_height_cm / person_height_px)

		# 7. Compute measurements in pixel space using mask bounds and BMI depth
		measurements_pixels = self.compute_measurements(
			front_pose=front_pose,
			person_bbox=person_bbox,
			mask=person_mask,
			person_weight_kg=person_weight_kg,
			person_height_cm=person_height_cm,
			gender=gender,
			v_factor=v_factor
		)

		# 8. Scale measurements into centimeters
		scaled_measurements = {}
		for key, value in measurements_pixels.items():
			if key != "pixel_height":
				scaled_measurements[key] = float(value * scale_cm_per_px)

		scaled_measurements["height"] = float(person_height_cm)

		if self._debug_visualization_enabled():
			draw_scaling_overlay(
				frame,
				person_bbox=person_bbox,
				scale_cm_per_px=scale_cm_per_px,
				estimated_height_cm=person_height_cm,
				filename="scaling_overlay.png",
			)

		# 9. Compute body fat from scaled waist and known height
		waist = scaled_measurements["waist"]
		body_fat = float(64 - (20 * waist / person_height_cm))

		# 10. Build feature vector
		features = self.build_feature_vector(
			gender,
			age,
			scaled_measurements,
			float(person_height_cm),
			body_fat,
		)

		# 11. Predict body type
		prediction = self.predict_body_type(features)

		return {
			"body_type": prediction["body_type"],
			"ayurvedic_type": prediction["ayurvedic_type"],
			"measurements": scaled_measurements,
			"meta": {
				"method": "height_reference",
				"scale_cm_per_px": scale_cm_per_px,
				"person_height_px": person_height_px,
				"tilt_angle_deg": math.degrees(tilt_angle),
				"detection_confidence": {
					"person": float(detection["person_confidence"]),
				},
			},
		}

	def validate_image(self, image):
		"""
		Validate image before processing.
		"""
		payload = validate_single_image(image)

		return payload

	def detect_person_only(self, image):
		"""
		Run person detection on frame.
		"""

		person_result = detect_person(
			image,
			self.person_model
		)

		if self._debug_visualization_enabled():
			draw_bbox(image, person_result["bbox"], filename="person_detection.png")

		return {
			"person_crop": person_result["crop"],
			"person_bbox": person_result["bbox"],
			"person_confidence": person_result["confidence"],
		}

	def validate_person_geometry(self, image, person_bbox):
		"""
		Reject frames that are likely to produce unstable scaling.
		"""

		if not is_bbox_fully_visible(person_bbox, image.shape):
			raise ValueError("Person is not fully visible. Please capture full body in frame")

	def extract_pose(self, person_crop):
		"""
		Extract pose keypoints from person crop.
		"""

		front_pose = extract_pose_keypoints(
			person_crop,
			self.pose_model,
		)

		if self._debug_visualization_enabled():
			draw_pose(person_crop, front_pose, filename="person_pose.png")

		return {
			"front_pose": front_pose,
		}

	def _detect_tilt(self, pose):
		"""
		Detect camera tilt using Nose-to-Hip verticality.
		Returns angle in radians.
		"""
		nose = pose["nose"]
		mid_hip = (
			(pose["left_hip"][0] + pose["right_hip"][0]) / 2.0,
			(pose["left_hip"][1] + pose["right_hip"][1]) / 2.0
		)
		
		dx = nose[0] - mid_hip[0]
		dy = nose[1] - mid_hip[1]
		
		angle = math.atan2(abs(dx), abs(dy))
		return angle

	def _get_vertical_correction_factor(self, tilt_angle):
		"""
		Returns 1/cos(theta) correction factor.
		"""
		return 1.0 / math.cos(tilt_angle)

	def _ellipse_circumference(self, width, depth):
		"""
		Approximate torso circumference from width/depth ellipse.
		"""

		a = max(float(width), 1.0) / 2.0
		b = max(float(depth), 1.0) / 2.0

		return math.pi * (3.0 * (a + b) - math.sqrt((3.0 * a + b) * (a + 3.0 * b)))

	def compute_measurements(self, front_pose, person_bbox, mask, person_weight_kg, person_height_cm, gender, v_factor):
		"""
		Compute measurements in pixel space from pose, person bbox, and generated mask.
		"""
		from measurement.torso_measurements import compute_torso_widths

		pixel_height = float(bbox_height(person_bbox)) * v_factor
		if pixel_height <= 0:
			raise ValueError("Invalid person height in pixels")

		torso_widths = compute_torso_widths(
			mask=mask,
			left_shoulder_x=min(front_pose["left_shoulder"][0], front_pose["right_shoulder"][0]),
			right_shoulder_x=max(front_pose["left_shoulder"][0], front_pose["right_shoulder"][0]),
			left_hip_x=min(front_pose["left_hip"][0], front_pose["right_hip"][0]),
			right_hip_x=max(front_pose["left_hip"][0], front_pose["right_hip"][0]),
			pose_shoulder_y=min(front_pose["left_shoulder"][1], front_pose["right_shoulder"][1]),
			pose_hip_y=max(front_pose["left_hip"][1], front_pose["right_hip"][1])
		)

		joint_shoulder_width = distance(front_pose["left_shoulder"], front_pose["right_shoulder"])
		joint_hip_width = distance(front_pose["left_hip"], front_pose["right_hip"])
		
		# Proportional fallbacks for robustness against poor segmentation
		# Shoulders: Deltoids usually extend ~15% past skeletal joints
		shoulder_width = max(torso_widths["mask_shoulder_width_pixels"], joint_shoulder_width * 1.25)
		
		# Chest: Usually 1.1x skeletal shoulder width
		chest_width = max(torso_widths["chest_width_pixels"], joint_shoulder_width * 1.15)
		
		# Waist: Adult male waist width is usually at least 65-70% of shoulder width
		# We also use the hip joint as a base reference.
		waist_width = max(torso_widths["waist_width_pixels"], shoulder_width * 0.7, joint_hip_width * 1.1)
		
		# Hips: Usually 1.2x skeletal hip distance OR at least as wide as the waist
		hip_width = max(torso_widths["hip_width_pixels"], waist_width, joint_hip_width * 1.2)

		# Clothing Factor subtraction (approx 1.5cm / 5%)
		clothing_reduction = 0.96
		chest_width *= clothing_reduction
		waist_width *= clothing_reduction
		hip_width *= clothing_reduction

		# BMI-based depth ratio calibration
		bmi = person_weight_kg / ((person_height_cm / 100.0) ** 2)
		
		# Gender-specific base depth calibration
		if gender == 1: # Male
			base_depth = 0.72
			bmi_mult = 0.015
		else: # Female
			base_depth = 0.65
			bmi_mult = 0.012
			
		# Allow leaner ratio for BMI < 22
		if bmi < 22:
			depth_ratio = base_depth - (22 - bmi) * 0.075
		else:
			depth_ratio = base_depth + (max(0, bmi - 22) * bmi_mult)
			
		depth_ratio = min(max(depth_ratio, 0.45), 0.95)

		chest = self._ellipse_circumference(chest_width, chest_width * (depth_ratio + 0.05)) # Chest is deeper
		waist = self._ellipse_circumference(waist_width, waist_width * depth_ratio)
		hips = self._ellipse_circumference(hip_width, hip_width * (depth_ratio - 0.03)) # Hips generally flatter

		measurements_pixels = {
			"pixel_height": float(pixel_height),
			"shoulder_width": shoulder_width,
			"chest": chest,
			"waist": waist,
			"hips": hips,
			"belly": waist * 1.1,
			"arm_length": compute_arm_length(front_pose),
			"shoulder_to_waist": compute_shoulder_to_waist(front_pose) * v_factor,
			"waist_to_knee": compute_waist_to_knee(front_pose) * v_factor,
			"leg_length": compute_leg_length_to_floor(front_pose, mask.shape[0]) * v_factor,
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
