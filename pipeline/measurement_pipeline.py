import os
import math

from utils.image_utils import validate_single_image
from utils.debug_visualization import (
	draw_bbox,
	draw_pose,
	draw_reference_overlay,
)
from measurement.feature_scaling import (
	build_feature_vector as build_ml_feature_vector,
)
from measurement.limb_measurements import (
	compute_arm_length,
	compute_leg_length,
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

	def _debug_visualization_enabled(self):
		return os.getenv("BODY_DEBUG_VIS", "0") == "1"

	def run(
		self,
		image,
		age,
		gender,
		door_real_height_cm=200.0,
		door_bbox_override=None,
	):
		"""
		Main pipeline execution for door-reference scaling.
		"""

		# 1. Validate input image
		payload = self.validate_image(image)
		frame = payload["image"]

		# 2. Detect person and door in the same frame
		detection = self.detect_entities(frame, door_bbox_override=door_bbox_override)
		person_bbox = detection["person_bbox"]
		door_bbox = detection["door_bbox"]

		# 3. Guardrails for common geometric failures
		self.validate_reference_geometry(frame, person_bbox, door_bbox)

		# 4. Extract pose from person crop
		pose_data = self.extract_pose(detection["person_crop"])
		front_pose = pose_data["front_pose"]

		# 5. Compute scale from known door height
		door_height_px = float(bbox_height(door_bbox))
		if door_height_px <= 0:
			raise ValueError("Invalid door height in pixels")

		scale_cm_per_px = float(door_real_height_cm / door_height_px)

		# 6. Compute measurements in pixel space
		measurements_pixels = self.compute_measurements(
			front_pose=front_pose,
			person_bbox=person_bbox,
		)

		# 7. Scale measurements into centimeters
		scaled_measurements = {}
		for key, value in measurements_pixels.items():
			if key != "pixel_height":
				scaled_measurements[key] = float(value * scale_cm_per_px)

		person_height_cm = float(measurements_pixels["pixel_height"] * scale_cm_per_px)
		scaled_measurements["height"] = person_height_cm

		if self._debug_visualization_enabled():
			draw_reference_overlay(
				frame,
				person_bbox=person_bbox,
				door_bbox=door_bbox,
				scale_cm_per_px=scale_cm_per_px,
				estimated_height_cm=person_height_cm,
				filename="reference_overlay.png",
			)

		# 8. Compute body fat from scaled waist and estimated height
		waist = scaled_measurements["waist"]
		body_fat = float(64 - (20 * waist / person_height_cm))

		# 9. Build feature vector
		features = self.build_feature_vector(
			gender,
			age,
			scaled_measurements,
			person_height_cm,
			body_fat,
		)

		# 10. Predict body type
		prediction = self.predict_body_type(features)

		return {
			"body_type": prediction["body_type"],
			"ayurvedic_type": prediction["ayurvedic_type"],
			"measurements": scaled_measurements,
			"meta": {
				"method": "door_reference",
				"scale_cm_per_px": scale_cm_per_px,
				"door_height_px": door_height_px,
				"person_height_px": float(measurements_pixels["pixel_height"]),
				"detection_confidence": {
					"person": float(detection["person_confidence"]),
					"door": float(detection["door_confidence"]),
				},
			},
		}

	def validate_image(self, image):
		"""
		Validate image before processing.
		"""
		payload = validate_single_image(image)

		return payload

	def detect_entities(self, image, door_bbox_override=None):
		"""
		Run person and door detection on same frame.
		"""

		person_result = detect_person(
			image,
			self.person_model
		)

		if door_bbox_override is not None:
			door_result = {
				"bbox": tuple(door_bbox_override),
				"crop": image[door_bbox_override[1]:door_bbox_override[3], door_bbox_override[0]:door_bbox_override[2]],
				"confidence": 1.0,
			}
		else:
			try:
				door_result = detect_door_heuristic(
					image
				)
			except ValueError as exc:
				raise ValueError(
					"Door detection failed. Provide --door-bbox x1,y1,x2,y2 or use an image with a clear full door"
				) from exc

		if self._debug_visualization_enabled():
			draw_bbox(image, person_result["bbox"], filename="person_detection.png")
			draw_bbox(image, door_result["bbox"], filename="door_detection.png")

		return {
			"person_crop": person_result["crop"],
			"person_bbox": person_result["bbox"],
			"person_confidence": person_result["confidence"],
			"door_bbox": door_result["bbox"],
			"door_confidence": door_result["confidence"],
		}

	def validate_reference_geometry(self, image, person_bbox, door_bbox):
		"""
		Reject frames that are likely to produce unstable scaling.
		"""

		if not is_bbox_fully_visible(door_bbox, image.shape):
			raise ValueError("Door is not fully visible. Please capture full door in frame")

		if not is_bbox_fully_visible(person_bbox, image.shape):
			raise ValueError("Person is not fully visible. Please capture full body in frame")

		frame_width = float(image.shape[1])
		person_center_x = bbox_center_x(person_bbox)
		door_center_x = bbox_center_x(door_bbox)

		if abs(person_center_x - door_center_x) > (0.35 * frame_width):
			raise ValueError("Person appears too far from door. Keep person and door on same plane")

		if bbox_height(door_bbox) <= bbox_height(person_bbox) * 0.5:
			raise ValueError("Detected door looks too small. Move camera or frame the full door")

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

	def _ellipse_circumference(self, width, depth):
		"""
		Approximate torso circumference from width/depth ellipse.
		"""

		a = max(float(width), 1.0) / 2.0
		b = max(float(depth), 1.0) / 2.0

		return math.pi * (3.0 * (a + b) - math.sqrt((3.0 * a + b) * (a + 3.0 * b)))

	def compute_measurements(self, front_pose, person_bbox):
		"""
		Compute measurements in pixel space from pose and person bbox.
		"""
		pixel_height = float(bbox_height(person_bbox))
		if pixel_height <= 0:
			raise ValueError("Invalid person height in pixels")

		shoulder_width = distance(
			front_pose["left_shoulder"],
			front_pose["right_shoulder"],
		)

		person_width = float(bbox_width(person_bbox))

		chest_width = max(shoulder_width * 1.18, person_width * 0.42)
		waist_width = max(chest_width * 0.88, person_width * 0.36)
		hip_width = max(waist_width * 1.06, person_width * 0.40)

		chest = self._ellipse_circumference(chest_width, chest_width * 0.68)
		waist = self._ellipse_circumference(waist_width, waist_width * 0.72)
		hips = self._ellipse_circumference(hip_width, hip_width * 0.74)

		if chest > 2 * shoulder_width:
			chest = shoulder_width * 1.5

		measurements_pixels = {
			"pixel_height": float(pixel_height),
			"shoulder_width": shoulder_width,
			"chest": chest,
			"waist": waist,
			"hips": hips,
			"belly": waist * 1.1,
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
