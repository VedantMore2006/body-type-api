from ultralytics import YOLO
import cv2
from vision.detection import detect_person

model = YOLO("yolov8n.pt")

img = cv2.imread("image.jpg")

result = detect_person(img, model)

print(result["bbox"])