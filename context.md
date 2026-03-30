## **API Architecture Overview (Version 2.0)**

This document provides a high-level summary of the current body measurement pipeline logic, used for both internal development and as a source of truth for the system's architecture.

---

### **1. Core Modalities & Logic**

The pipeline estimates 3D body measurements from a **single 2D front-facing image** by leveraging physical constants (Height/Weight) to solve for the missing 3D dimensions.

1.  **Validation**: [utils/image_utils.py](utils/image_utils.py) — Validates resolution and format.
2.  **Detection**: [vision/detection.py](vision/detection.py) — YOLOv8 (yolov8n.pt) identifies the person and crops the frame to the subject.
3.  **Pose Estimation**: [vision/pose.py](vision/pose.py) — YOLOv8-pose (yolov8n-pose.pt) extracts 17 skeletal landmarks for anatomical alignment.
4.  **Segmentation**: [vision/segmentation.py](vision/segmentation.py) — MediaPipe `SelfieSegmentation` generates a binary silhouette mask.
5.  **Scaling**: [pipeline/measurement_pipeline.py](pipeline/measurement_pipeline.py) — Directly scales pixels to real-world centimeters using the user-provided **Ground Truth Height**.
6.  **Depth Estimation**: [pipeline/measurement_pipeline.py](pipeline/measurement_pipeline.py) — Calculates a dynamic **Depth Ratio** based on the user's **Body Mass Index (BMI)** to estimate front-to-back thickness.
7.  **Circumferences**: [pipeline/measurement_pipeline.py](pipeline/measurement_pipeline.py) — Applies the Ramanujan ellipse approximation using measured width (from mask) and calculated depth.
8.  **Prediction**: [pipeline/measurement_pipeline.py](pipeline/measurement_pipeline.py) — RandomForestClassifier predicts Western body type; mapped to Ayurvedic Dosha (Vata, Pitta, Kapha).

---

### **2. Input Requirements**

- **Image**: Single front-facing color image.
- **Physical Data**: `person_height_cm` (Ground Truth), `person_weight_kg` (Ground Truth), `age`, `gender` (0=F, 1=M).

### **3. Key Engineering Components**

| Component | Purpose | Key File |
|-----------|---------|----------|
| **YOLOv8** | Fast Person & Pose detection | [vision/pose.py](vision/pose.py) |
| **MediaPipe** | High-contrast silhouette mask | [vision/segmentation.py](vision/segmentation.py) |
| **Torso Scanning** | Horizontal width profile extraction | [measurement/torso_measurements.py](measurement/torso_measurements.py) |
| **BMI Depth Logic**| Dynamic thickness prediction | [pipeline/measurement_pipeline.py](pipeline/measurement_pipeline.py) |
| **Random Forest** | Categorical Body Type prediction | [models/model_loader.py](models/model_loader.py) |

---

### **4. Notable Differences from Previous Baseline**

- **NO DOOR/A4 REFERENCE**: The system was successfully migrated from a reference-object baseline (Door/A4 sheet) to a **Ground Truth Reference** system, significantly reducing perspective error.
- **SINGLE IMAGE**: Eliminated the need for a side-profile image by implementing a parametric depth model based on BMI.
- **SEGMENTATION-FIRST WIDTH**: Widths are now measured from the true mask boundaries at anatomical rows, rather than using heuristic multipliers of the shoulders.

### **5. Limitations**

- **Perspective Sensitivity**: Assumes the camera is level and at a medium distance (3–5 meters).
- **Clothing Geometry**: Success depends on the visibility of the body silhouette (avoiding very baggy clothing).
- **Volume Approximation**: Depth estimation is a statistical approximation for a single-view system.

---

For deeper technical detail, model weights, and measurement formulas, refer to [docs/technical_specification.md](docs/technical_specification.md).
───┐
    │ Compute Body Composition             │
    │ (Body Fat, Feature Vector)           │
    └────────┬─────────────────────────────┘
             ↓
    ┌──────────────────────────────────────┐
    │ Body Type Prediction (SVM)           │
    │ Ectomorph / Mesomorph / Endomorph   │
    │ → Vata / Pitta / Kapha (Ayurvedic)  │
    └──────────────────────────────────────┘
             ↓
    Output: {body_type, ayurvedic_type, measurements}
```

---

## **KEY INSIGHTS**

1. **Measurements are fundamentally 2D pixel-based:** Width from frontal silhouette, depth from side silhouette, both sampled at anatomically-consistent row indices.
2. **Scaling hinges entirely on user-provided height:** There is **no active calibration object** (README mentions A4 sheet but code shows none); height is external input.
3. **Ellipse approximation for circumferences:** Chest/waist/hip circumferences computed from 2D width (front) and depth (side) via Ramanujan's ellipse formula.
4. **Keypoint skeletal measurements supplementary:** Arm, leg, shoulder-to-waist distances use 17-point pose skeleton directly; independent of silhouette.
5. **No camera modeling:** No intrinsic/extrinsic matrix, no distortion correction; assumes orthographic projection and perpendicular geometry.