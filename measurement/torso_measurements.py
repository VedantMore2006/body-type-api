import numpy as np
from scipy.ndimage import gaussian_filter1d


def compute_width_profile(mask, left_shoulder_x, right_shoulder_x):
    """
    Compute horizontal width of the body at each row.
    Scan only the band between the shoulders to exclude arms.
    """

    left_limit = max(int(left_shoulder_x - 40), 0)
    right_limit = min(int(right_shoulder_x + 40), mask.shape[1])

    height = mask.shape[0]
    widths = []

    for y in range(height):

        row = mask[y]

        region = row[left_limit:right_limit]

        body_pixels = np.where(region == 1)[0]

        if len(body_pixels) == 0:
            widths.append(0)
            continue

        left = body_pixels[0]
        right = body_pixels[-1]

        widths.append(right - left)

    widths = np.array(widths)

    widths = gaussian_filter1d(widths, sigma=3)

    return widths


def detect_torso_measurements(width_profile, pose_shoulder_y, pose_hip_y):

    height = len(width_profile)
    if height == 0:
        raise ValueError("Invalid width profile")

    torso_start = int(pose_shoulder_y)
    torso_end = int(pose_hip_y)

    # Clamp to valid array bounds.
    torso_start = min(max(torso_start, 0), height - 1)
    torso_end = min(max(torso_end, torso_start + 1), height)

    torso = width_profile[torso_start:torso_end]
    if torso.size == 0:
        chest_y = waist_y = hip_y = height // 2
        return chest_y, waist_y, hip_y

    waist_index = int(np.argmin(torso))
    waist_y = torso_start + waist_index

    upper_torso = width_profile[torso_start:waist_y]
    if upper_torso.size == 0:
        chest_y = torso_start
    else:
        chest_y = torso_start + int(np.argmax(upper_torso))

    lower_torso = width_profile[waist_y:torso_end]
    if lower_torso.size == 0:
        hip_y = waist_y
    else:
        hip_y = waist_y + int(np.argmax(lower_torso))

    chest_y = int(min(max(chest_y, 0), height - 1))
    waist_y = int(min(max(waist_y, 0), height - 1))
    hip_y = int(min(max(hip_y, 0), height - 1))

    return chest_y, waist_y, hip_y


def compute_torso_widths(mask, left_shoulder_x, right_shoulder_x, pose_shoulder_y, pose_hip_y):

    shoulder_width_px = max(float(right_shoulder_x - left_shoulder_x), 1.0)
    max_torso_width = 1.4 * shoulder_width_px

    width_profile = compute_width_profile(mask, left_shoulder_x, right_shoulder_x)

    chest_y, waist_y, hip_y = detect_torso_measurements(width_profile, pose_shoulder_y, pose_hip_y)

    chest_width = min(float(width_profile[chest_y]), max_torso_width)
    waist_width = min(float(width_profile[waist_y]), max_torso_width)
    hip_width = min(float(width_profile[hip_y]), max_torso_width)

    return {
        "chest_width_pixels": chest_width,
        "waist_width_pixels": waist_width,
        "hip_width_pixels": hip_width
    }
