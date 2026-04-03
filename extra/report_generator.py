import cv2
import numpy as np
import os

class ReportGenerator:
    def __init__(self, output_path="Final_Result_Infographic.png"):
        self.output_path = output_path
        # Define color palette (Premium Dark Mode)
        self.COLORS = {
            "bg": (28, 28, 28),
            "accent": (75, 181, 67), # Green for success/health
            "text": (240, 240, 240),
            "subtext": (180, 180, 180),
            "border": (60, 60, 60),
            "vata": (255, 180, 100), # Light Blue (Wait, RGB vs BGR: BGR Blue is 255,180,100)
            "pitta": (100, 100, 255), # Soft Red (BGR Red is 100,100,255)
            "kapha": (100, 255, 180), # Soft Green
        }

    def generate(self, measurements, body_type, ayurvedic_type, person_height_cm, person_weight_kg):
        """
        Generate a premium 1200x800 infographic report.
        """
        width, height = 1200, 800
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = self.COLORS["bg"]

        # Draw Background Gradient/Texture
        cv2.rectangle(canvas, (0, 0), (width, 150), (40, 40, 40), -1)
        cv2.line(canvas, (0, 150), (width, 150), self.COLORS["border"], 2)

        # Header
        cv2.putText(canvas, "AI BODY ARCHITECT", (50, 60), cv2.FONT_HERSHEY_DUPLEX, 1, self.COLORS["subtext"], 1, cv2.LINE_AA)
        cv2.putText(canvas, "Ayurvedic Health Analysis", (50, 110), cv2.FONT_HERSHEY_TRIPLEX, 1.5, self.COLORS["text"], 2, cv2.LINE_AA)

        # Body Type & Dosha
        dosha_key = ayurvedic_type.lower()
        dosha_color = self.COLORS.get(dosha_key, self.COLORS["accent"])
        
        cv2.putText(canvas, "Body Type:", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLORS["subtext"], 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{body_type}", (50, 270), cv2.FONT_HERSHEY_DUPLEX, 1.8, self.COLORS["text"], 2, cv2.LINE_AA)

        cv2.putText(canvas, "Ayurvedic Dosha:", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLORS["subtext"], 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{ayurvedic_type.upper()}", (50, 410), cv2.FONT_HERSHEY_TRIPLEX, 2.2, dosha_color, 3, cv2.LINE_AA)

        # Measurement Grid
        start_x = 600
        start_y = 220
        row_h = 60
        
        items = [
            ("Height", f"{person_height_cm} cm"),
            ("Weight", f"{person_weight_kg} kg"),
            ("Shoulder", f"{measurements['shoulder_width']:.1f} cm"),
            ("Chest", f"{measurements['chest']:.1f} cm"),
            ("Waist", f"{measurements['waist']:.1f} cm"),
            ("Hips", f"{measurements['hips']:.1f} cm"),
            ("Belly", f"{measurements['belly']:.1f} cm"),
            ("Arm Length", f"{measurements['arm_length']:.1f} cm"),
            ("Leg Length", f"{measurements['leg_length']:.1f} cm"),
        ]

        cv2.putText(canvas, "PHYSICAL PARAMETERS", (start_x, 190), cv2.FONT_HERSHEY_DUPLEX, 0.7, self.COLORS["accent"], 1, cv2.LINE_AA)
        
        for i, (label, val) in enumerate(items):
            y = start_y + i * row_h
            cv2.putText(canvas, label, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLORS["subtext"], 1, cv2.LINE_AA)
            cv2.putText(canvas, val, (start_x + 300, y), cv2.FONT_HERSHEY_DUPLEX, 0.8, self.COLORS["text"], 1, cv2.LINE_AA)
            cv2.line(canvas, (start_x, y + 15), (start_x + 500, y + 15), (50, 50, 50), 1)

        # Footer / Seal
        cv2.rectangle(canvas, (50, 700), (400, 710), dosha_color, -1)
        cv2.putText(canvas, "VERIFIED BY ANTIGRAVITY AI", (50, 750), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS["subtext"], 1, cv2.LINE_AA)

        cv2.imwrite(self.output_path, canvas)
        print(f"Final Premium Report saved to {self.output_path}")
