from ultralytics import YOLO
import joblib
from vision.segmentation import HumanSegmenter


class ModelLoader:

    def __init__(self):
        self.person_model = None
        self.pose_model = None
        self.bodytype_model = None
        self.label_encoder = None
        self.segmenter = None

    def load_models(self):

        if self.person_model is None:
            print("Loading YOLO person model...")
            self.person_model = YOLO("yolov8n.pt")

        if self.pose_model is None:
            print("Loading YOLO pose model...")
            self.pose_model = YOLO("yolov8n-pose.pt")

        if self.bodytype_model is None:
            print("Loading body type model...")
            self.bodytype_model = joblib.load("bodytype_model.pkl")

        if self.label_encoder is None:
            print("Loading label encoder...")
            self.label_encoder = joblib.load("label_encoder.pkl")

        if self.segmenter is None:
            print("Loading segmenter...")
            self.segmenter = HumanSegmenter()

        return {
            "person_model": self.person_model,
            "pose_model": self.pose_model,
            "bodytype_model": self.bodytype_model,
            "label_encoder": self.label_encoder,
            "segmenter": self.segmenter
        }