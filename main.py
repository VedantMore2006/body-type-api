import argparse
import json
import os
from pathlib import Path

import cv2
import os

os.environ['QT_QPA_PLATFORM'] = 'xcb'

ROOT = Path(__file__).resolve().parent


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


def run_pipeline(front_image_path: str, side_image_path: str, age: float, gender: float, height: float):
    """Run the body measurement pipeline and return API-compatible output."""
    from models.model_loader import ModelLoader
    from pipeline.measurement_pipeline import MeasurementPipeline

    front_path = _resolve_image_path(front_image_path)
    side_path = _resolve_image_path(side_image_path)

    front_img = _load_image(front_path)
    side_img = _load_image(side_path)

    models = ModelLoader().load_models()
    pipeline = MeasurementPipeline(models)

    return pipeline.run(
        front_image=front_img,
        side_image=side_img,
        age=age,
        gender=gender,
        height=height,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Body measurement pipeline CLI")
    parser.add_argument("--front-image", default="test/front1.jpg", help="Front image path")
    parser.add_argument("--side-image", default="test/side1.jpg", help="Side image path")
    parser.add_argument("--gender", type=float, default=1, help="Gender (0 or 1)")
    parser.add_argument("--age", type=float, default=20, help="Age in years")
    parser.add_argument("--height", type=float, default=174, help="Total height in centimeters")
    parser.add_argument("--debug-vis", type=int, choices=[0, 1], default=1, help="Save debug visualizations (1=on, 0=off)")
    return parser.parse_args()


def main():
    from utils.debug_visualization import clear_debug_folder

    args = parse_args()
    os.environ["BODY_DEBUG_VIS"] = str(args.debug_vis)
    clear_debug_folder()
    result = run_pipeline(
        front_image_path=args.front_image,
        side_image_path=args.side_image,
        age=args.age,
        gender=args.gender,
        height=args.height,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()