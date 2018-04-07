import numpy as np


def wrap_in_ndarray(value):
    """Wraps the argument in a numpy.ndarray.
    
       If value is a scalar, it is converted in a list first.
       If value is array-like, the shape is conserved.
    
    """

    if hasattr(value, "__len__"):
        return np.array(value)
    else:
        return np.array([value])
