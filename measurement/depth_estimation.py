import numpy as np
import math


def compute_depth_profile(mask):
    """
    Compute body depth profile from side silhouette mask.
    """

    height = mask.shape[0]

    depths = []

    for y in range(height):

        row = mask[y]

        body_pixels = np.where(row == 1)[0]

        if len(body_pixels) == 0:
            depths.append(0)
            continue

        left = body_pixels[0]
        right = body_pixels[-1]

        depths.append(right - left)

    return np.array(depths)


def compute_torso_depths(side_mask, chest_y, waist_y, hip_y):

    depth_profile = compute_depth_profile(side_mask)

    chest_depth = depth_profile[chest_y]
    waist_depth = depth_profile[waist_y]
    hip_depth = depth_profile[hip_y]

    return {
        "chest_depth_pixels": chest_depth,
        "waist_depth_pixels": waist_depth,
        "hip_depth_pixels": hip_depth
    }


def ellipse_circumference(width, depth):

    a = width / 2
    b = depth / 2

    C = math.pi * (3*(a+b) - math.sqrt((3*a + b)*(a + 3*b)))

    return C


def compute_torso_circumferences(widths, depths):

    chest = ellipse_circumference(
        widths["chest_width_pixels"],
        depths["chest_depth_pixels"]
    )

    waist = ellipse_circumference(
        widths["waist_width_pixels"],
        depths["waist_depth_pixels"]
    )

    hips = ellipse_circumference(
        widths["hip_width_pixels"],
        depths["hip_depth_pixels"]
    )

    return {
        "chest_pixels": chest,
        "waist_pixels": waist,
        "hips_pixels": hips
    }
