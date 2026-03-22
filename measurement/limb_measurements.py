import numpy as np


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def compute_arm_length(pose):

    left = distance(pose["left_shoulder"], pose["left_elbow"]) + \
           distance(pose["left_elbow"], pose["left_wrist"])

    right = distance(pose["right_shoulder"], pose["right_elbow"]) + \
            distance(pose["right_elbow"], pose["right_wrist"])

    return max(left, right)


def compute_leg_length(pose):

    left = distance(pose["left_hip"], pose["left_knee"]) + \
           distance(pose["left_knee"], pose["left_ankle"])

    right = distance(pose["right_hip"], pose["right_knee"]) + \
            distance(pose["right_knee"], pose["right_ankle"])

    return max(left, right)


def compute_shoulder_to_waist(pose):

    shoulder_mid = (
        (pose["left_shoulder"][0] + pose["right_shoulder"][0]) / 2,
        (pose["left_shoulder"][1] + pose["right_shoulder"][1]) / 2
    )

    hip_mid = (
        (pose["left_hip"][0] + pose["right_hip"][0]) / 2,
        (pose["left_hip"][1] + pose["right_hip"][1]) / 2
    )

    return distance(shoulder_mid, hip_mid)


def compute_waist_to_knee(pose):

    hip_mid = (
        (pose["left_hip"][0] + pose["right_hip"][0]) / 2,
        (pose["left_hip"][1] + pose["right_hip"][1]) / 2
    )

    knee_mid = (
        (pose["left_knee"][0] + pose["right_knee"][0]) / 2,
        (pose["left_knee"][1] + pose["right_knee"][1]) / 2
    )

    return distance(hip_mid, knee_mid)
