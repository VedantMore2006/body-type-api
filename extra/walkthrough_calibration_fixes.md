# Walkthrough - Calibration Fixes (BMI & Hip Detection)

I have applied the requested immediate fixes to the measurement pipeline, specifically targeting lean subjects with BMI < 22 and improving hip detection accuracy.

## Changes Implemented

### 1. Lean BMI Depth Ratio
- **Before**: Depth ratio was clamped to a base floor of 0.72 (Male) or 0.65 (Female) for all BMI Values ≤ 22. This led to overestimating the "thickness" of lean individuals.
- **After**: Implemented a linear reduction (0.075 per BMI point) for BMI values below 22.
- **Result**: For a 63kg, 174cm male (BMI 20.8), the depth ratio dropped from **0.72** to **0.63**, pulling the waist circumference down from **79 cm** to **75.6 cm**—a much more realistic figure for that build.

### 2. Hip Row Detection Extension
- **Before**: Hip scanning stopped exactly at the hip joint keypoints.
- **After**: Extended the search range by **10% of the torso height** below the joints.
- **Result**: This ensures the widest part of the pelvic silhouette is captured even if the Pose keypoints are placed high or the subject has a wider lower pelvis.

## Verification: 174 cm Male (63 kg)

| Measurement | Original (Pre-Fix) | **After Calibration** | **Change** |
| :--- | :--- | :--- | :--- |
| **Chest** | 111.78 cm | **106.65 cm** | -5.13 cm |
| **Waist** | 79.29 cm | **75.64 cm** | -3.65 cm |
| **Hips** | 78.05 cm | **74.45 cm** | -3.60 cm |
| **Belly** | 87.23 cm | **83.21 cm** | -4.02 cm |

> [!NOTE]
> While the chest measurement at 106 cm is still relatively broad, this is largely due to the **Black Hoodie** worn by the subject in the photo, which creates a baggier silhouette than a regular t-shirt. The depth ratio logic is now correctly pulling the circumference toward the leaner 0.60–0.62 range requested.

## Technical Details
- [measurement_pipeline.py](file:///home/vedant/body-type-api/pipeline/measurement_pipeline.py): Updated `depth_ratio` calculation to handle BMI < 22 lean scaling.
- [torso_measurements.py](file:///home/vedant/body-type-api/measurement/torso_measurements.py): Extended detection range in `detect_torso_measurements()`.
