def compute_scale(real_height, pixel_height):
    """
    Compute pixel-to-real-world scaling factor.
    """

    if pixel_height == 0:
        raise ValueError("Pixel height cannot be zero")

    scale = real_height / pixel_height

    return scale


def scale_measurements(measurements, scale):

    scaled = {}

    for key, value in measurements.items():
        scaled[key] = value * scale

    return scaled


def compute_body_fat(waist, height):

    body_fat = 64 - (20 * waist / height)

    return body_fat


def build_feature_vector(
    gender,
    age,
    measurements,
    total_height,
    body_fat
):

    features = [
        gender,
        age,

        measurements["shoulder_width"],
        measurements["chest"],
        measurements["belly"],
        measurements["waist"],
        measurements["hips"],

        measurements["arm_length"],
        measurements["shoulder_to_waist"],
        measurements["waist_to_knee"],
        measurements["leg_length"],

        total_height,
        body_fat
    ]

    return features
