import cv2
import numpy as np


def validate_image(image):
    """
    Validate decoded image.
    """

    if image is None:
        raise ValueError("Image decoding failed")

    if not isinstance(image, np.ndarray):
        raise ValueError("Invalid image format")

    if image.size == 0:
        raise ValueError("Empty image data")

    height, width = image.shape[:2]

    if height == 0 or width == 0:
        raise ValueError("Image has invalid dimensions")

    if height < 200 or width < 200:
        raise ValueError("Image resolution too small")

    return True


def validate_single_image(image):
    """
    Validate and return a single image payload.
    """

    validate_image(image)

    return {
        "image": image
    }


def validate_dual_images(front_image, side_image):
    """
    Validate both front and side images.
    """

    validate_image(front_image)
    validate_image(side_image)

    return {
        "front": front_image,
        "side": side_image
    }
