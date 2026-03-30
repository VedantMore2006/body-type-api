# Technical Specification: Body Measurement & Ayurvedic Mapping

This document provides a detailed understanding of the current measurement modalities, the AI models used, input/output requirements, and the scientific logic behind the system.

## 1. Modalities & Architecture

The system follows a multi-stage computer vision pipeline to estimate 3D body measurements from a single front-facing 2D image.

### A. Person Detection
- **Model:** YOLOv8 Nano (`yolov8n.pt`)
- **Purpose:** To localize the human subject in the frame and generate a tight bounding box crop. This increases the accuracy of subsequent stages by focusing standard processing on the relevant pixels.

### B. Pose Estimation
- **Model:** YOLOv8 Nano Pose (`yolov8n-pose.pt`)
- **Purpose:** To extract 17 key anatomical landmarks (shoulders, hips, knees, ankles, etc.).
- **Usage:**
  - **Limb Measurements:** Directly calculates distances for arm length and leg length.
  - **Anatomical Alignment:** Keypoints identify the exact Y-coordinates (vertical rows) where the chest, waist, and hips are located, regardless of pose or clothing.

### C. Body Segmentation
- **Model:** MediaPipe `SelfieSegmentation` (Model Selection 1 - Large)
- **Purpose:** To generate a high-contrast binary silhouette mask (segmentation mask).
- **Utility:** Provides the basis for "True Width" extraction by allowing the system to scan horizontal pixel counts at specific anatomical rows determined by the pose model.

### D. Body Type Classification
- **Model:** RandomForestClassifier (`bodytype_model.pkl`)
- **Input:** A 13-element feature vector (gender, age, 11 scaled measurements).
- **Goal:** Predicts Western body types (Ectomorph, Mesomorph, Endomorph).
- **Ayurvedic Mapping:**
  - Ectomorph → **Vata**
  - Mesomorph → **Pitta**
  - Endomorph → **Kapha**

---

## 2. Scientific Logic: Physical Scaling & Depth

Unlike older systems that require a door or an A4 sheet for reference, this system uses the **Ground Truth Height** and **Weight** provided by the user for superior accuracy.

### Direct Height Scaling
The system calculates a precise scaling factor (cm per pixel) by comparing the user's provided ground truth height to the vertical pixel height of the detected person:
`Scale (cm/px) = User_Height_CM / Detected_Pixel_Height`

### BMI-Based Dynamic Depth
A core challenge of 2D imagery is the "hidden depth" (front-to-back thickness). We solve this using a dynamic density mapping:
1. **BMI Calculation:** `Weight_kg / (Height_m²)`
2. **Dynamic Depth Ratio:**
   `Depth_Ratio = 0.65 + (max(0, BMI - 22) * 0.012)`
   - This ratio adapts the torso thickness estimate based on weight. As BMI increases, the "depth factor" increases dynamically.
3. **Circumference Formula:**
   We use the Ramanujan ellipse approximation to calculate girths:
   $$C \approx \pi [3(a+b) - \sqrt{(3a+b)(a+3b)}]$$
   - *$a$ = half of the measured pixel width (from segmentation mask)*
   - *$b$ = half of the estimated pixel depth (width × depth_ratio)*

---

## 3. Input & Output Requirements

### Required Inputs
- **Image**: Single front-facing image (subject's full body visible, neutral pose).
- **Height (cm)**: User's actual height.
- **Weight (kg)**: User's actual weight.
- **Age**: User's age.
- **Gender**: 0 (Female) or 1 (Male).

### System Outputs
- **Western Body Type**: Morphological category.
- **Ayurvedic Type**: Corresponding Dosha.
- **Body Measurements**:
  - Arm, Leg, and Torso lengths.
  - Chest, Waist, Hips, and Belly circumferences.
  - Shoulder width.
- **Metadata**: Scaling factors, pixel heights, and detection confidence levels.

---

## 4. Accuracy & Performance Factors

The accuracy of this model is influenced by:
1. **Clothing:** Form-fitting clothing provides the most accurate silhouette. Extremely baggy clothing will lead to wider circumference estimates.
2. **Camera Distance/Angle:** The camera should be at approximately waist height and parallel to the floor to minimize perspective distortion.
3. **Pose Consistency:** The subject should stand straight in a neutral "A-pose" or similar for optimal landmark detection.
4. **Resolution:** High-resolution images allow for more precise edge detection in the segmentation stage.

> [!IMPORTANT]
> The current accuracy is estimated at ±3% for length measurements and ±6% for circumferences when the subject follows the pose and clothing guidelines.
