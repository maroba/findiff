import numpy as np
from findiff.diff import FinDiff, Coefficient
from findiff.util import wrap_in_ndarray


class VectorOperator(object):

    def __init__(self, **kwargs):

        if "h" in kwargs:
            self.h = wrap_in_ndarray(kwargs.pop("h"))
            self.ndims = len(self.h)
            self.components = [FinDiff((k, self.h[k]), **kwargs) for k in range(self.ndims)]
        if "coords" in kwargs:
            coords = kwargs.pop("coords")
            if isinstance(coords, np.ndarray):
                shape = coords.shape
                if len(shape) > 1:
                    ndims = shape[0]
                else:
                    ndims = 1
            else:
                ndims = len(coords)
            self.ndims = ndims

            self.components = [FinDiff((k,), coords=self.coords, **kwargs) for k in range(self.ndims)]


class Gradient(VectorOperator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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


class Divergence(VectorOperator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, f):

        if not isinstance(f, np.ndarray) and not isinstance(f, list):
            raise TypeError("Function to differentiate must be numpy.ndarray or list of numpy.ndarrays")

        if len(f.shape) != self.ndims + 1 and f.shape[0] != self.ndims:
            raise ValueError("Divergence can only be applied to vector functions of the same dimension")

        result = np.zeros(f.shape[1:])

        for k in range(self.ndims):
            result += self.components[k](f[k])

        return result


class Curl(VectorOperator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.ndims != 3:
            raise ValueError("Curl operation is only defined in 3 dimensions. {} were given.".format(self.ndims))

    def __call__(self, f):

        if not isinstance(f, np.ndarray) and not isinstance(f, list):
            raise TypeError("Function to differentiate must be numpy.ndarray or list of numpy.ndarrays")

        if len(f.shape) != self.ndims + 1 and f.shape[0] != self.ndims:
            raise ValueError("Curl can only be applied to vector functions of the three dimensions")

        result = np.zeros(f.shape)

        result[0] += self.components[1](f[2]) - self.components[2](f[1])
        result[1] += self.components[2](f[0]) - self.components[0](f[2])
        result[2] += self.components[0](f[1]) - self.components[1](f[0])

        return result
