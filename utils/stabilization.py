import numpy as np


def median_filter_measurements(measurement_list):
    """
    Compute median values across multiple measurement frames.
    """

    keys = measurement_list[0].keys()

    stabilized = {}

    for key in keys:

        values = [m[key] for m in measurement_list]

        stabilized[key] = float(np.median(values))

    return stabilized
