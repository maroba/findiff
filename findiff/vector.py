import numpy as np
from findiff.diff import FinDiff
from findiff.util import wrap_in_ndarray


class Gradient(object):

    def __init__(self, **kwargs):

        if "dims" in kwargs:
            raise ValueError("dims cannot be specified for Gradient")

        if "h" in kwargs:
            h = wrap_in_ndarray(kwargs["h"])
            ndims = len(h)
        if "coords" in kwargs:
            coords = kwargs["coords"]
            if isinstance(coords, np.ndarray):
                shape = coords.shape
                if len(shape) > 1:
                    ndims = shape[0]
                else:
                    ndims = 1
            else:
                ndims = len(coords)

        self.components = [FinDiff(dims=[k], **kwargs) for k in range(ndims)]
        self.ndims = len(self.components)

    def __call__(self, f):

        if not isinstance(f, np.ndarray):
            raise TypeError("Function to differentiate must be numpy.ndarray")

        if len(f.shape) != self.ndims:
            raise ValueError("Gradients can only be applied to scalar functions")

        result = []
        for k in range(self.ndims):
            d_dxk =  self.components[k]
            result.append(d_dxk(f))

        return np.array(result)
