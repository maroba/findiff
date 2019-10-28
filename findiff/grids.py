import numpy as np


class Grid(object):
    pass


class UniformGrid(Grid):

    def __init__(self, shape, spac, center=None):
        self.shape = shape
        self.ndims = len(shape)
        if not hasattr(spac, '__len__'):
            self.spac = spac,
        else:
            self.spac = spac

        if center is None:
            self.center = np.zeros(self.ndims)
        else:
            assert len(center) == self.ndims
            self.center = np.array(center)

    def spacing(self, axis):
        return self.spac[axis]
