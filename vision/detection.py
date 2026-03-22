import cv2


def detect_person(image, model):
    """
    Detect person using YOLOv8 and return bounding box and cropped image.
    """

    results = model(image)

    person_bbox = None

    for r in results:
        boxes = r.boxes.xyxy

        if boxes is None or len(boxes) == 0:
            raise ValueError("No person detected")

        # Take first detected person
        x1, y1, x2, y2 = map(int, boxes[0][:4])

        person_bbox = (x1, y1, x2, y2)

        break

    if person_bbox is None:
        raise ValueError("Person detection failed")

    x1, y1, x2, y2 = person_bbox

    person_crop = image[y1:y2, x1:x2]

    return {
        "bbox": person_bbox,
        "crop": person_crop
    }
