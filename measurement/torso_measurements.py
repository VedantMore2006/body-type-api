import numpy as np
from scipy.ndimage import gaussian_filter1d


def compute_width_profile(mask, left_shoulder_x, right_shoulder_x, left_hip_x, right_hip_x):
    """
    Compute horizontal width of the torso at each row.
    Excludes arms by picking only the segment that contains or is closest to the spine line.
    """
    height = mask.shape[0]
    
    # Define spinal centerline row-by-row for better tracking if person is leaning
    shoulder_y = (mask.shape[0] // 2) # Default fallback
    shoulder_center_x = (left_shoulder_x + right_shoulder_x) / 2.0
    hip_center_x = (left_hip_x + right_hip_x) / 2.0
    
    widths = []

    for y in range(height):
        # Linear interpolation for spine center at row y
        # (Assuming torso is between shoulder_y and hip_y)
        # For simplicity, we'll use a fixed corridor based on the average center
        center_x = (shoulder_center_x + hip_center_x) / 2.0
        
        row = mask[y]
        
        # Find contiguous segments of body pixels in the whole row
        diff = np.diff(np.concatenate(([0], row, [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        if len(starts) == 0:
            widths.append(0)
            continue
            
        # Find the segment that overlaps with the center_x
        # or find the one whose center is closest to center_x
        segment_centers = (starts + ends) / 2.0
        segment_widths = ends - starts
        
        # We want the torso, which is usually the largest segment near the center
        # Filter for segments near the center (within 500px)
        near_center = np.where(np.abs(segment_centers - center_x) < 500)[0]
        
        if len(near_center) == 0:
            # Pick the segment closest to center regardless of distance
            central_idx = np.argmin(np.abs(segment_centers - center_x))
            torso_width = segment_widths[central_idx]
        else:
            # Of the segments near center, pick the largest one (the torso)
            torso_width = np.max(segment_widths[near_center])
             
        widths.append(torso_width)

    widths = np.array(widths)
    # Low sigma to preserve accurate contour but smooth row-to-row jitter
    widths = gaussian_filter1d(widths, sigma=2)

    return widths


def detect_torso_measurements(width_profile, pose_shoulder_y, pose_hip_y):

    height = len(width_profile)
    if height == 0:
        raise ValueError("Invalid width profile")

    torso_start = int(pose_shoulder_y)
    # Extend hip scan range ~10% below the hip joint to capture pelvic silhouette
    torso_end = int(pose_hip_y + 0.1 * (pose_hip_y - pose_shoulder_y))

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


def compute_mask_shoulder_width(mask, pose_shoulder_y):
    """
    Search mask at shoulder level to find the true outer edge of deltoids.
    """
    y = int(pose_shoulder_y)
    y = min(max(y, 0), mask.shape[0] - 1)
    
    row = mask[y]
    body_pixels = np.where(row == 1)[0]
    
    if len(body_pixels) == 0:
        return 0.0
    
    return float(body_pixels[-1] - body_pixels[0])


def compute_torso_widths(mask, left_shoulder_x, right_shoulder_x, left_hip_x, right_hip_x, pose_shoulder_y, pose_hip_y):

    # Using mask-based shoulder width as a more realistic reference than joint-to-joint.
    mask_shoulder_width = compute_mask_shoulder_width(mask, pose_shoulder_y)
    
    # We still want some sanity check to avoid arm-bleeding, but 1.4 is too low if joints are narrow.
    # We'll use 1.6x joints OR the mask-based width itself as a conservative upper bound.
    joint_width = max(float(right_shoulder_x - left_shoulder_x), 1.0)
    max_torso_width = max(1.6 * joint_width, mask_shoulder_width * 1.25)

    width_profile = compute_width_profile(mask, left_shoulder_x, right_shoulder_x, left_hip_x, right_hip_x)

    chest_y, waist_y, hip_y = detect_torso_measurements(width_profile, pose_shoulder_y, pose_hip_y)

    chest_width = min(float(width_profile[chest_y]), max_torso_width)
    waist_width = min(float(width_profile[waist_y]), max_torso_width)
    hip_width = min(float(width_profile[hip_y]), max_torso_width)

    return {
        "chest_width_pixels": chest_width,
        "waist_width_pixels": waist_width,
        "hip_width_pixels": hip_width,
        "mask_shoulder_width_pixels": mask_shoulder_width
    }
