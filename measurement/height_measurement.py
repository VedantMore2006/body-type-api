import numpy as np


def compute_pixel_height(binary_mask):
    """
    Compute pixel height from segmentation mask.
    """

    rows = np.any(binary_mask, axis=1)
    if not np.any(rows):
        raise ValueError("No body detected during processing. Ensure the image is clear and the person is visible.")

    top = np.argmax(rows)
    bottom = len(rows) - np.argmax(rows[::-1]) - 1

    pixel_height = bottom - top
    if pixel_height <= 0:
        raise ValueError("Invalid person height computed from the image.")

    return pixel_height, top, bottom
