# Body Measurement & Ayurvedic Body Type API

This project estimates body measurements and predicts Ayurvedic body types (Vata, Pitta, Kapha) from a single front-facing image using the user's known height and weight as calibration references.

## Current System (Version 2.0 - Scientific Scaling)

- **Input**: One front-facing image, height (cm), weight (kg), age, and gender.
- **Person Detection**: YOLOv8 (`yolov8n.pt`) for localization.
- **Pose Estimation**: YOLOv8-pose (`yolov8n-pose.pt`) for skeletal landmarking.
- **Body Segmentation**: MediaPipe `SelfieSegmentation` for pixel-perfect silhouette extraction.
- **Scaling Logic**: `cm_per_pixel = user_height_cm / silhouette_pixel_height`.
- **Dynamic Depth**: Torso circumferences (chest, waist, hips) are calculated using the Ramanujan ellipse approximation, where depth is dynamically estimated based on the user's **Body Mass Index (BMI)** and gender proportions.
- **Body Type Classification**: A Random Forest model predicts Western body types (Ectomorph, Mesomorph, Endomorph) which are then mapped to Ayurvedic Doshas.

## Why This Approach

By using the user's actual height for scaling, we eliminate the perspective distortion errors inherent in door-reference or A4-sheet methods. The integration of weight and BMI allows for realistic 3D volume estimation (depth) from a single 2D image.

## Getting Started

### Prerequisites

- Python 3.9+
- Dependencies: `pip install -r requirements.txt` (includes `mediapipe`, `ultralytics`, `scipy`, etc.)
- Weights: `yolov8n.pt`, `yolov8n-pose.pt`, `bodytype_model.pkl`, `label_encoder.pkl`

### Command Line Interface

```bash
python main.py \
   --image test/front1.png \
   --person-height-cm 180 \
   --person-weight-kg 75 \
   --age 25 \
   --gender 1 \
   --debug-vis 1
```

## Output

The pipeline returns a JSON object:

- `body_type`: (Ectomorph, Mesomorph, Endomorph)
- `ayurvedic_type`: (Vata, Pitta, Kapha)
- `measurements`: (height, shoulder_width, chest, waist, hips, belly, arm_length, leg_length, etc. in cm)
- `meta`: (scaling factor, pixel metrics, detection confidence)

When `--debug-vis 1` is enabled, check the `debug/` folder for:
- `scaling_overlay.png`: Visual verification of the person height scaling.
- `segmentation_mask.png`: The silhouette used for width extraction.
- `person_pose.png`: YOLO skeletal keypoints.
- `torso_rows.png`: Highlights where the chest, waist, and hip rows were detected.

## Technical Documentation

For a detailed breakdown of the models, mathematical formulas, and accuracy considerations, see [docs/technical_specification.md](docs/technical_specification.md).
