import itertools

import numpy as np
from scipy import sparse

from findiff.coefs import coefficients_non_uni, coefficients
from findiff.grids import GridAxis, EquidistantAxis, NonEquidistantAxis
from findiff.utils import long_indices_as_ndarray, to_long_index


def build_differentiator(order: int, axis: GridAxis, acc):
    if isinstance(axis, EquidistantAxis):
        if not axis.periodic:
            return _FinDiffUniform(axis.dim, order, axis.spacing, acc)
        else:
            return _FinDiffUniformPeriodic(axis.dim, order, axis.spacing, acc)
    elif isinstance(axis, NonEquidistantAxis):
        if not axis.periodic:
            return _FinDiffNonUniform(axis.dim, order, axis.coords, acc)
        else:
            raise NotImplementedError("Periodic nonuniform axes not yet implemented")
    else:
        raise TypeError("Unknown axis type.")


class _FinDiffBase:

    def __init__(self, axis, order):
        self.axis = axis
        self.order = order

    def validate_f(self, f):
        try:
            f.shape[self.axis]
        except AttributeError as err:
            raise ValueError(
                "Diff objects can only be applied to arrays or evaluated(!) functions returning arrays"
            ) from err

    def apply_to_array(self, yd, y, weights, off_slices, ref_slice, dim):
        """Applies the finite differences only to slices along a given axis"""

        ndims = len(y.shape)

        all = slice(None, None, 1)

        ref_multi_slice = [all] * ndims
        ref_multi_slice[dim] = ref_slice

        for w, s in zip(weights, off_slices):
            off_multi_slice = [all] * ndims
            off_multi_slice[dim] = s
            if abs(1 - w) < 1.0e-14:
                yd[tuple(ref_multi_slice)] += y[tuple(off_multi_slice)]
            else:
                yd[tuple(ref_multi_slice)] += w * y[tuple(off_multi_slice)]

    def shift_slice(self, sl, off, max_index):

        if sl.start + off < 0 or sl.stop + off > max_index:
            raise IndexError("Shift slice out of bounds")

        return slice(sl.start + off, sl.stop + off, sl.step)


class _FinDiffUniform(_FinDiffBase):

    def __init__(self, axis, order, spacing, acc):
        super().__init__(axis, order)
        self.spacing = spacing
        self.acc = acc
        coef_schemes = coefficients(self.order, acc)
        self.forward = coef_schemes["forward"]
        self.backward = coef_schemes["backward"]
        self.center = coef_schemes["center"]

    def __call__(self, f):
        self.validate_f(f)
        npts = f.shape[self.axis]
        weights = self.center["coefficients"]
        offsets = self.center["offsets"]

        num_bndry_points = len(weights) // 2
        ref_slice = slice(num_bndry_points, npts - num_bndry_points, 1)
        off_slices = [
            self.shift_slice(ref_slice, offsets[k], npts) for k in range(len(offsets))
        ]

        fd = np.zeros_like(f)

        self.apply_to_array(fd, f, weights, off_slices, ref_slice, self.axis)

        weights = self.forward["coefficients"]
        offsets = self.forward["offsets"]

        ref_slice = slice(0, num_bndry_points, 1)
        off_slices = [
            self.shift_slice(ref_slice, offsets[k], npts) for k in range(len(offsets))
        ]

        self.apply_to_array(fd, f, weights, off_slices, ref_slice, self.axis)

        weights = self.backward["coefficients"]
        offsets = self.backward["offsets"]

        ref_slice = slice(npts - num_bndry_points, npts, 1)
        off_slices = [
            self.shift_slice(ref_slice, offsets[k], npts) for k in range(len(offsets))
        ]

        self.apply_to_array(fd, f, weights, off_slices, ref_slice, self.axis)

        h_inv = 1.0 / self.spacing**self.order
        return fd * h_inv

    def matrix(self, shape):

        h = self.spacing

        ndims = len(shape)
        siz = np.prod(shape)
        long_indices_nd = long_indices_as_ndarray(shape)

        axis, order = self.axis, self.order
        mat = sparse.lil_matrix((siz, siz))

        for scheme in ["center", "forward", "backward"]:

            offsets_1d = getattr(self, scheme)["offsets"]
            coeffs = getattr(self, scheme)["coefficients"]

            # translate offsets of given scheme to long format
            offsets_long = []
            for o_1d in offsets_1d:
                o_nd = np.zeros(ndims)
                o_nd[axis] = o_1d
                o_long = to_long_index(o_nd, shape)
                offsets_long.append(o_long)

            # determine points where to evaluate current scheme in long format
            nside = len(self.center["coefficients"]) // 2
            if scheme == "center":
                multi_slice = [slice(None, None)] * ndims
                multi_slice[axis] = slice(nside, -nside)
                Is = long_indices_nd[tuple(multi_slice)].reshape(-1)
            elif scheme == "forward":
                multi_slice = [slice(None, None)] * ndims
                multi_slice[axis] = slice(0, nside)
                Is = long_indices_nd[tuple(multi_slice)].reshape(-1)
            else:
                multi_slice = [slice(None, None)] * ndims
                multi_slice[axis] = slice(-nside, None)
                Is = long_indices_nd[tuple(multi_slice)].reshape(-1)

            for o, c in zip(offsets_long, coeffs):
                v = c / h**order
                mat[Is, Is + o] = v

        return mat


class _FinDiffUniformPeriodic(_FinDiffBase):

    def __init__(self, axis, order, spacing, acc):
        super().__init__(axis, order)
        self.spacing = spacing
        self.acc = acc
        self.coefs = coefficients(self.order, acc)["center"]

    def __call__(self, f):
        self.validate_f(f)
        fd = np.zeros_like(f)
        for off, coef in zip(self.coefs["offsets"], self.coefs["coefficients"]):
            fd += coef * np.roll(f, -off, axis=self.axis)
        h_inv = 1.0 / self.spacing**self.order
        return fd * h_inv

    def matrix(self, shape):
        h = self.spacing

        ndims = len(shape)
        siz = np.prod(shape)
        long_indices_nd = long_indices_as_ndarray(shape)

        axis, order = self.axis, self.order
        mat = sparse.lil_matrix((siz, siz))

        offsets = self.coefs["offsets"]
        coefs = self.coefs["coefficients"]

        multi_slice = [slice(None, None)] * ndims
        Is = long_indices_nd[tuple(multi_slice)].reshape(-1)

        idxs_short = [np.arange(n) for n in shape]

        for o, c in zip(offsets, coefs):
            v = c / h**order

            idxs_short[self.axis] = np.roll(np.arange(shape[self.axis]), -o)
            grid = np.meshgrid(*idxs_short, indexing="ij")
            index_tuples = np.stack(grid, axis=-1).reshape(-1, ndims)

            Is_off = np.ravel_multi_index(index_tuples.T, shape)

            mat[Is, Is_off] = v

        return mat


class _FinDiffNonUniform(_FinDiffBase):
    def __init__(self, axis, order, coords, acc):
        super().__init__(axis, order)
        self.coords = coords
        self.acc = acc
        self.coef_list = []
        for i in range(len(self.coords)):
            self.coef_list.append(coefficients_non_uni(order, self.acc, self.coords, i))

    def __call__(self, y):
        """The core function to take a partial derivative on a non-uniform grid"""

        order, dim = self.order, self.axis
        yd = np.zeros_like(y)

        ndims = len(y.shape)
        multi_slice = [slice(None, None)] * ndims
        ref_multi_slice = [slice(None, None)] * ndims

        for i, x in enumerate(self.coords):

            coefs = self.coef_list[i]
            weights = coefs["coefficients"]
            offsets = coefs["offsets"]
            ref_multi_slice[dim] = i

            for off, w in zip(offsets, weights):
                multi_slice[dim] = i + off
                yd[tuple(ref_multi_slice)] += w * y[tuple(multi_slice)]

        return yd

    def matrix(self, shape):

        coords = self.coords

        siz = np.prod(shape)
        long_inds = np.arange(siz).reshape(shape)
        short_inds = [np.arange(shape[k]) for k in range(len(shape))]
        short_inds = list(itertools.product(*short_inds))

        coef_dicts = []
        for i in range(len(coords)):
            coef_dicts.append(coefficients_non_uni(self.order, self.acc, coords, i))

        mat = sparse.lil_matrix((siz, siz))

        for base_ind_long, base_ind_short in enumerate(short_inds):
            cd = coef_dicts[base_ind_short[self.axis]]
            cs, os = cd["coefficients"], cd["offsets"]
            for c, o in zip(cs, os):
                off_short = np.zeros(len(shape), dtype=int)
                off_short[self.axis] = int(o)
                off_ind_short = np.array(base_ind_short, dtype=int) + off_short
                off_long = long_inds[tuple(off_ind_short)]

                mat[base_ind_long, off_long] += c

        return mat
