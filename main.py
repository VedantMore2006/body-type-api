import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import os

os.environ['QT_QPA_PLATFORM'] = 'xcb'

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _resolve_image_path(path_str: str) -> Path:
    """Resolve image path and support .jpg -> .jpeg fallback for defaults."""
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT / path

    if path.exists():
        return path

    # Keep user-requested defaults while supporting existing test assets.
    if path.suffix.lower() == ".jpg":
        jpeg_path = path.with_suffix(".jpeg")
        if jpeg_path.exists():
            return jpeg_path

    raise FileNotFoundError(f"Image not found: {path}")


def _load_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    return image


def _parse_bbox_arg(bbox_text: str | None):
    if not bbox_text:
        return None

    values = [item.strip() for item in bbox_text.split(",")]
    if len(values) != 4:
        raise ValueError("--door-bbox must be in format x1,y1,x2,y2")

    x1, y1, x2, y2 = map(int, values)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("--door-bbox coordinates are invalid")

    return (x1, y1, x2, y2)


def run_pipeline(image_path: str, age: float, gender: float, door_height_cm: float, door_bbox=None):
    """Run the body measurement pipeline and return API-compatible output."""
    from models.model_loader import ModelLoader
    from pipeline.measurement_pipeline import MeasurementPipeline

    source_path = _resolve_image_path(image_path)

    image = _load_image(source_path)

    models = ModelLoader().load_models()
    pipeline = MeasurementPipeline(models)

    return pipeline.run(
        image=image,
        age=age,
        gender=gender,
        door_real_height_cm=door_height_cm,
        door_bbox_override=door_bbox,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Body measurement pipeline CLI")
    parser.add_argument("--image", default="test/front1.jpg", help="Image path containing person and door")
    parser.add_argument("--gender", type=float, default=1, help="Gender (0 or 1)")
    parser.add_argument("--age", type=float, default=20, help="Age in years")
    parser.add_argument("--door-height-cm", type=float, default=200.0, help="Known real-world door height in centimeters")
    parser.add_argument("--door-bbox", default=None, help="Optional manual door bbox as x1,y1,x2,y2")
    parser.add_argument("--debug-vis", type=int, choices=[0, 1], default=1, help="Save debug visualizations (1=on, 0=off)")
    return parser.parse_args()


def main():
    from utils.debug_visualization import clear_debug_folder

    args = parse_args()
    door_bbox = _parse_bbox_arg(args.door_bbox)
    os.environ["BODY_DEBUG_VIS"] = str(args.debug_vis)
    clear_debug_folder()
    result = run_pipeline(
        image_path=args.image,
        age=args.age,
        gender=args.gender,
        door_height_cm=args.door_height_cm,
        door_bbox=door_bbox,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()