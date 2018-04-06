import numpy as np


def wrap_in_ndarray(value):

    if hasattr(value, "__len__"):
        return np.array(value)
    else:
        return np.array([value])
