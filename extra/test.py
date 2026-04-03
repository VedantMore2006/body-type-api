import argparse

import cv2
from ultralytics import YOLO

from vision.detection import detect_door_heuristic, detect_person


def main():
	parser = argparse.ArgumentParser(description="Quick local detection check")
	parser.add_argument("--image", required=True, help="Path to image containing person and door")
	args = parser.parse_args()

	image = cv2.imread(args.image, cv2.IMREAD_COLOR)
	if image is None:
		raise ValueError(f"Failed to load image: {args.image}")

	model = YOLO("yolov8n.pt")

	person = detect_person(image, model)
	door = detect_door_heuristic(image)

	print("person_bbox:", person["bbox"], "conf:", round(person["confidence"], 4))
	print("door_bbox:", door["bbox"], "conf:", round(door["confidence"], 4))


if __name__ == "__main__":
	main()