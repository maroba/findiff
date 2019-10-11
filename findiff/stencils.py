from itertools import product
from copy import deepcopy
import operator
import numpy as np
from .coefs import coefficients


class Stencil(object):

    def __init__(self, shape, axis, order, h, acc, old_stl=None):
        self.shape = shape
        self.axis = axis
        self.order = order
        self.h = h
        self.acc = acc
        self.char_pts = self._det_characteristic_points()
        if old_stl:
            self.data = old_stl.data
        else:
            self.data = {}

        self._create_stencil()

    def apply(self, u, idx0):

        if not hasattr(idx0, '__len__'):
            idx0 = (idx0, )

        typ = []
        for axis in range(len(self.shape)):
            if idx0[axis] == 0:
                typ.append('L')
            elif idx0[axis] == self.shape[axis] - 1:
                typ.append('H')
            else:
                typ.append('C')
        typ = tuple(typ)

        stl = self.data[typ]

        idx0 = np.array(idx0)
        du = 0.
        for o, c in stl.items():
            idx = idx0 + o
            du += c * u[tuple(idx)]

        return du

    def apply_all(self, u):

        assert self.shape == u.shape

        ndims = len(u.shape)
        if ndims == 1:
            indices = list(range(len(u)))
        else:
            axes_indices = []
            for axis in range(ndims):
                axes_indices.append(list(range(u.shape[axis])))

            axes_indices = tuple(axes_indices)
            indices = list(product(*axes_indices))

        du = np.zeros_like(u)

        for idx in indices:
            du[idx] = self.apply(u, idx)

        return du

    def for_point(self, idx):
        typ = self.type_for_point(idx)
        return self.data[typ]

    def type_for_point(self, idx):
        typ = []
        for axis in range(len(idx)):
            if idx[axis] == 0:
                typ.append('L')
            elif idx[axis] == self.shape[axis] - 1:
                typ.append('H')
            else:
                typ.append('C')
        return tuple(typ)

    def _create_stencil(self):

        ndim = len(self.shape)
        data = self.data
        smap = {'L': 'forward', 'C': 'center', 'H': 'backward'}

        axis, order, h = self.axis, self.order, self.h

        for pt in self.char_pts:
            scheme = smap[pt[axis]]
            coefs = coefficients(order, self.acc)[scheme]

            if pt in data:
                lstl = data[pt]
            else:
                lstl = {}

            for off, c in zip(coefs['offsets'], coefs['coefficients']):
                long_off = [0] * ndim
                long_off[axis] += off
                long_off = tuple(long_off)

                if long_off in lstl:
                    lstl[long_off] += c / h ** order
                else:
                    lstl[long_off] = c / h ** order
            data[pt] = lstl

    def _det_characteristic_points(self):
        shape = self.shape
        ndim = len(shape)
        typ = [("L", "C", "H")]*ndim
        return product(*typ)

    def __str__(self):
        s = ""
        for typ, stl in self.data.items():
            s += str(typ) + ":\t" + str(stl) + "\n"
        return s

    def _binaryop(self, other, op):
        stl = deepcopy(self)
        assert stl.shape == other.shape

        for char_pt, single_stl in stl.data.items():
            other_single_stl = other.data[char_pt]
            for o, c in other_single_stl.items():
                if o in single_stl:
                    single_stl[o] = op(single_stl[o], c)
                else:
                    single_stl[o] = op(0, c)

        return stl

    def __add__(self, other):
        return self._binaryop(other, operator.__add__)

    def __sub__(self, other):
        return self._binaryop(other, operator.__sub__)