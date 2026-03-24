import cv2
import numpy as np


PERSON_CLASS_ID = 0


def bbox_height(bbox):
    x1, y1, x2, y2 = bbox
    _ = x1
    _ = x2
    return max(0, int(y2 - y1))


def bbox_width(bbox):
    x1, y1, x2, y2 = bbox
    _ = y1
    _ = y2
    return max(0, int(x2 - x1))


def bbox_center_x(bbox):
    x1, _, x2, _ = bbox
    return float(x1 + x2) / 2.0


def is_bbox_fully_visible(bbox, image_shape, margin=2):
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox

    return (
        x1 > margin
        and y1 > margin
        and x2 < (width - margin)
        and y2 < (height - margin)
    )


def _clip_bbox(bbox, image_shape):
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(x1 + 1, min(int(x2), width))
    y2 = max(y1 + 1, min(int(y2), height))

    return (x1, y1, x2, y2)


def detect_person(image, model, min_confidence=0.25):
    """
    Detect the most confident person box using YOLO class filtering.
    """

    results = model(image, verbose=False)

    best_bbox = None
    best_conf = -1.0

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        confidences = r.boxes.conf.cpu().numpy()

        for idx, cls_id in enumerate(classes):
            conf = float(confidences[idx])
            if cls_id != PERSON_CLASS_ID or conf < min_confidence:
                continue

            bbox = tuple(map(int, boxes[idx][:4]))
            bbox = _clip_bbox(bbox, image.shape)
            if conf > best_conf:
                best_conf = conf
                best_bbox = bbox

    if best_bbox is None:
        raise ValueError("No person detected")

    x1, y1, x2, y2 = best_bbox
    person_crop = image[y1:y2, x1:x2]

    return {
        "bbox": best_bbox,
        "crop": person_crop,
        "confidence": float(best_conf),
    }


def detect_door_heuristic(image):
    """
    Detect a door-like vertical rectangle using OpenCV contour heuristics.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 45, 140)
    edge_kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, edge_kernel, iterations=2)
    edges = cv2.erode(edges, edge_kernel, iterations=1)

    # Also use threshold-based regions so clean rectangular doors survive when
    # contour edges are fragmented.
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
        iterations=1,
    )

    merged = cv2.bitwise_or(edges, thresh)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No door-like region found")

    img_h, img_w = image.shape[:2]
    img_area = float(img_h * img_w)

    best_bbox = None
    best_score = -1.0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 35 or h < 140:
            continue

        area = float(w * h)
        area_ratio = area / max(img_area, 1.0)
        if area_ratio < 0.008:
            continue

        aspect_ratio = float(h) / float(max(w, 1))
        if aspect_ratio < 1.8 or aspect_ratio > 8.0:
            continue

        contour_area = float(cv2.contourArea(contour))
        rectangularity = contour_area / max(area, 1.0)
        if rectangularity < 0.06:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        quad_bonus = 0.25 if len(approx) in (4, 5) else 0.0

        vertical_coverage = float(h) / float(max(img_h, 1))
        center_x = x + (w / 2.0)
        center_bias = 1.0 - min(abs(center_x - (img_w / 2.0)) / (img_w / 2.0), 1.0)

        score = (
            (area_ratio * 1.8)
            + min(aspect_ratio / 4.0, 1.0)
            + min(rectangularity, 1.0)
            + (vertical_coverage * 1.2)
            + (center_bias * 0.35)
            + quad_bonus
        )
        if score > best_score:
            best_score = score
            best_bbox = (x, y, x + w, y + h)

    if best_bbox is None:
        raise ValueError("Door detection failed")

    best_bbox = _clip_bbox(best_bbox, image.shape)
    x1, y1, x2, y2 = best_bbox
    door_crop = image[y1:y2, x1:x2]

    confidence = float(min(max(best_score / 3.0, 0.0), 1.0))

    return {
        "bbox": best_bbox,
        "crop": door_crop,
        "confidence": confidence,
    }
