# Body Measurement Pipeline (Door Reference Baseline)

This project estimates body measurements from a single image using a known-height door as the reference object.

## Current Baseline

- Input: one image containing both person and door.
- Person detection: YOLOv8 (`yolov8n.pt`).
- Door detection: OpenCV contour-based vertical rectangle heuristic.
- Scale: `cm_per_pixel = door_real_height_cm / door_height_px`.
- Height estimate: `person_height_cm = person_height_px * cm_per_pixel`.
- Additional measurements: pose-based limb distances and heuristic torso circumferences, all scaled by the same factor.

## Why This Baseline

The goal is to ship a simple and fast reference-object approach first, then iterate on robustness.

## Limitations

- Door detector is heuristic and may fail in cluttered scenes.
- Perspective and depth mismatch can distort scale.
- Camera tilt can bias vertical pixel measurements.
- Torso circumferences are approximations in this phase.

## Run Locally

### Prerequisites

- Python environment with dependencies from `requirements.txt`
- Model files:
   - `yolov8n.pt`
   - `yolov8n-pose.pt`
   - `bodytype_model.pkl`
   - `label_encoder.pkl`

### Command

```bash
python main.py \
   --image test/front1.jpg \
   --door-height-cm 200 \
   --age 24 \
   --gender 1 \
   --debug-vis 0
```

If automatic door detection fails, provide the door region manually:

```bash
python main.py \
   --image test/front1.jpg \
   --door-height-cm 200 \
   --door-bbox 120,40,310,980 \
   --age 24 \
   --gender 1 \
   --debug-vis 0
```

## Output

The pipeline returns:

- `body_type`
- `ayurvedic_type`
- `measurements` (including estimated `height` in cm)
- `meta` (method, scale, pixel heights, detection confidences)

## Next Up

- Add a trained door detector as a fallback/upgrade path.
- Add frame quality confidence and stricter rejection rules.
- Add multi-frame smoothing for stability.
