import numpy as np


def extract_pose_keypoints(image, pose_model):
    """
    Run YOLOv8 pose estimation and extract keypoints required
    for limb measurements.
    """

    results = pose_model(image)

    for r in results:

        keypoints = r.keypoints.xy.cpu().numpy()

        if keypoints is None or len(keypoints) == 0:
            raise ValueError("No pose detected")

        k = keypoints[0]

        pose_data = {
            "left_shoulder": k[5],
            "right_shoulder": k[6],

            "left_elbow": k[7],
            "right_elbow": k[8],

            "left_wrist": k[9],
            "right_wrist": k[10],

            "left_hip": k[11],
            "right_hip": k[12],

            "left_knee": k[13],
            "right_knee": k[14],

            "left_ankle": k[15],
            "right_ankle": k[16],
        }

        return pose_data

    raise ValueError("Pose extraction failed")
