import os

import cv2
import matplotlib.pyplot as plt

DEBUG_DIR = "/home/vedant/body-measurement-api/debug"


def clear_debug_folder():
    """Remove prior debug outputs so each run writes a clean, replaced set."""
    os.makedirs(DEBUG_DIR, exist_ok=True)
    for name in os.listdir(DEBUG_DIR):
        path = os.path.join(DEBUG_DIR, name)
        if os.path.isfile(path):
            os.remove(path)


def _save(filename, img):
    """Save an OpenCV image to the debug folder, overwriting any previous file."""
    os.makedirs(DEBUG_DIR, exist_ok=True)
    path = os.path.join(DEBUG_DIR, filename)
    cv2.imwrite(path, img)


def _show(window_name, img):
    """Attempt an on-screen display; silently skip in headless environments."""
    try:
        cv2.imshow(window_name, img)
        cv2.waitKey(1)
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass


def draw_bbox(image, bbox, filename="person_detection.png"):
    """Draw person detection bounding box."""
    x1, y1, x2, y2 = bbox

    img = image.copy()
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    _save(filename, img)
    _show("Person Detection", img)


def draw_reference_overlay(
    image,
    person_bbox,
    door_bbox,
    scale_cm_per_px,
    estimated_height_cm,
    filename="reference_overlay.png",
):
    """Draw person + door boxes and reference scaling summary on one frame."""
    img = image.copy()

    px1, py1, px2, py2 = person_bbox
    dx1, dy1, dx2, dy2 = door_bbox

    cv2.rectangle(img, (px1, py1), (px2, py2), (60, 220, 60), 3)
    cv2.rectangle(img, (dx1, dy1), (dx2, dy2), (60, 160, 255), 3)

    cv2.putText(img, "Person", (px1, max(py1 - 10, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 220, 60), 2)
    cv2.putText(img, "Door", (dx1, max(dy1 - 10, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 160, 255), 2)

    door_height_px = max(dy2 - dy1, 1)
    person_height_px = max(py2 - py1, 1)

    summary_lines = [
        f"scale: {scale_cm_per_px:.4f} cm/px",
        f"door height: {door_height_px} px",
        f"person height: {person_height_px} px",
        f"estimated height: {estimated_height_cm:.1f} cm",
    ]

    x0 = 14
    y0 = 28
    line_h = 24

    panel_w = 430
    panel_h = 18 + line_h * len(summary_lines)
    cv2.rectangle(img, (x0 - 8, y0 - 18), (x0 - 8 + panel_w, y0 - 18 + panel_h), (0, 0, 0), -1)
    cv2.rectangle(img, (x0 - 8, y0 - 18), (x0 - 8 + panel_w, y0 - 18 + panel_h), (230, 230, 230), 1)

    for idx, line in enumerate(summary_lines):
        y = y0 + (idx * line_h)
        cv2.putText(img, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

    _save(filename, img)
    _show("Reference Overlay", img)


def show_mask(mask, filename="segmentation_mask.png"):
    """Visualize binary segmentation mask."""
    vis = (mask * 255).astype("uint8")

    _save(filename, vis)
    _show("Segmentation Mask", vis)


def overlay_mask(image, mask, filename="mask_overlay.png"):
    """Overlay binary mask on image."""
    colored = image.copy()
    colored[mask == 1] = [0, 255, 0]
    blended = cv2.addWeighted(image, 0.7, colored, 0.3, 0)

    _save(filename, blended)
    _show("Mask Overlay", blended)


def save_segmentation_with_bbox(image, mask, filename="segmentation_with_bbox.png"):
    """Save segmentation overlay with a tight mask bounding box and labels."""
    colored = image.copy()
    colored[mask == 1] = [0, 255, 0]
    blended = cv2.addWeighted(image, 0.7, colored, 0.3, 0)

    mask_u8 = (mask.astype("uint8") * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        cv2.rectangle(blended, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(blended, "Segmentation BBox", (x, max(y - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    cv2.putText(blended, "Green: Segmentation", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    _save(filename, blended)
    _show("Segmentation With BBox", blended)


def draw_pose(image, pose, filename="pose.png"):
    """Draw pose keypoints with labels."""
    img = image.copy()

    for name, point in pose.items():
        x, y = int(point[0]), int(point[1])
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    _save(filename, img)
    _show("Pose", img)


def draw_torso_lines(image, chest_y, waist_y, hip_y, filename="torso_rows.png"):
    """Draw horizontal torso measurement rows."""
    img = image.copy()

    h, w, _ = img.shape
    _ = h

    cv2.line(img, (0, int(chest_y)), (w, int(chest_y)), (255, 0, 0), 2)
    cv2.line(img, (0, int(waist_y)), (w, int(waist_y)), (0, 255, 0), 2)
    cv2.line(img, (0, int(hip_y)),   (w, int(hip_y)),   (0, 0, 255), 2)

    cv2.putText(img, "Chest (Blue)", (8, max(int(chest_y) - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
    cv2.putText(img, "Waist (Green)", (8, max(int(waist_y) - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.putText(img, "Hips (Red)", (8, max(int(hip_y) - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    _save(filename, img)
    _show("Torso Rows", img)


def plot_width_profile(width_profile, filename="width_profile.png"):
    """Plot torso width profile curve."""
    fig, ax = plt.subplots()
    ax.plot(width_profile)
    ax.set_title("Torso Width Profile")
    ax.set_xlabel("Body Height")
    ax.set_ylabel("Width")

    os.makedirs(DEBUG_DIR, exist_ok=True)
    path = os.path.join(DEBUG_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
