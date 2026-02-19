import numpy as np
from scipy import sparse

from findiff.backend import add_at, get_namespace
from findiff.coefs import coefficients, calc_coefs_non_uni_batched
from findiff.compact import _CompactDiffUniformPeriodic, _CompactDiffUniformNonPeriodic
from findiff.grids import EquidistantAxis, GridAxis, NonEquidistantAxis
from findiff.utils import (
    get_list_of_multiindex_tuples,
    get_long_indices_for_all_grid_points_as_1d_array,
    get_long_indices_for_all_grid_points_as_ndarray,
    to_long_index,
)


def build_differentiator(order: int, axis: GridAxis, acc, scheme=None):
    if scheme:
        if isinstance(axis, EquidistantAxis):
            if axis.periodic:
                return _CompactDiffUniformPeriodic(
                    axis.dim, order, axis.spacing, scheme
                )
            else:
                return _CompactDiffUniformNonPeriodic(
                    axis.dim, order, axis.spacing, scheme
                )
        raise NotImplementedError("This is not yet implemented for compact schemes.")

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

    def guard_valid_target(self, f):
        try:
            f.shape[self.axis]
        except AttributeError as err:
            raise ValueError(
                "Diff objects can only be applied to arrays or evaluated(!) functions returning arrays"
            ) from err

        if np.issubdtype(f.dtype, np.integer):
            f = f.astype(np.float64)
        return f

    def apply_to_array(self, yd, y, weights, off_slices, ref_slice, dim):
        """Applies the finite differences only to slices along a given axis.

        Returns the updated output array (required for immutable backends
        like JAX).
        """

        ndims = len(y.shape)

        all = slice(None, None, 1)

        ref_multi_slice = [all] * ndims
        ref_multi_slice[dim] = ref_slice

        for w, s in zip(weights, off_slices):
            off_multi_slice = [all] * ndims
            off_multi_slice[dim] = s
            if abs(1 - w) < 1.0e-14:
                yd = add_at(yd, tuple(ref_multi_slice), y[tuple(off_multi_slice)])
            else:
                yd = add_at(yd, tuple(ref_multi_slice), w * y[tuple(off_multi_slice)])
        return yd

    def shift_slice(self, sl, off, max_index):

        if sl.start + off < 0 or sl.stop + off > max_index:
            raise IndexError("Shift slice out of bounds")

        return slice(sl.start + off, sl.stop + off, sl.step)

    def matrix(self, shape):
        siz = np.prod(shape)
        mat = sparse.lil_matrix((siz, siz))
        self.write_matrix_entries(mat, shape)
        return sparse.csr_matrix(mat)

    def write_matrix_entries(self, mat, shape):
        raise NotImplementedError


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
        f = self.guard_valid_target(f)
        xp = get_namespace(f)

        npts = f.shape[self.axis]
        fd = xp.zeros_like(f)
        num_bndry_points = len(self.center["coefficients"]) // 2

        fd = self._apply_central_coefs(f, fd, npts, num_bndry_points)

        fd = self._apply_forward_coefs(f, fd, npts, num_bndry_points)

        fd = self._apply_backward_coefs(f, fd, npts, num_bndry_points)

        h_inv = 1.0 / self.spacing**self.order
        return fd * h_inv

    def _apply_backward_coefs(self, f, fd, npts, num_bndry_points):
        weights = self.backward["coefficients"]
        offsets = self.backward["offsets"]
        ref_slice = slice(npts - num_bndry_points, npts, 1)
        off_slices = [
            self.shift_slice(ref_slice, offsets[k], npts) for k in range(len(offsets))
        ]
        return self.apply_to_array(fd, f, weights, off_slices, ref_slice, self.axis)

    def _apply_forward_coefs(self, f, fd, npts, num_bndry_points):
        weights = self.forward["coefficients"]
        offsets = self.forward["offsets"]
        ref_slice = slice(0, num_bndry_points, 1)
        off_slices = [
            self.shift_slice(ref_slice, offsets[k], npts) for k in range(len(offsets))
        ]
        return self.apply_to_array(fd, f, weights, off_slices, ref_slice, self.axis)

    def _apply_central_coefs(self, f, fd, npts, num_bndry_points):
        weights = self.center["coefficients"]
        offsets = self.center["offsets"]
        ref_slice = slice(num_bndry_points, npts - num_bndry_points, 1)
        off_slices = [
            self.shift_slice(ref_slice, offsets[k], npts) for k in range(len(offsets))
        ]
        return self.apply_to_array(fd, f, weights, off_slices, ref_slice, self.axis)

    def write_matrix_entries(self, mat, shape):
        long_indices_nd = get_long_indices_for_all_grid_points_as_ndarray(shape)
        for scheme in ["center", "forward", "backward"]:

            offsets_long = self._convert_1D_offsets_to_long_indices(
                self.axis, getattr(self, scheme)["offsets"], shape
            )

            multi_slice = self._get_multislice_for_scheme(self.axis, scheme, shape)
            Is = long_indices_nd[tuple(multi_slice)].reshape(-1)

            coefs = getattr(self, scheme)["coefficients"]
            for o, c in zip(offsets_long, coefs):
                v = c / self.spacing**self.order
                mat[Is, Is + o] = v

    def _get_multislice_for_scheme(self, axis, scheme, shape):
        ndims = len(shape)
        multi_slice = [slice(None, None)] * ndims
        nside = len(self.center["coefficients"]) // 2
        if scheme == "center":
            multi_slice[axis] = slice(nside, -nside)
        elif scheme == "forward":
            multi_slice[axis] = slice(0, nside)
        else:
            multi_slice[axis] = slice(-nside, None)
        return multi_slice

    def _convert_1D_offsets_to_long_indices(self, axis, offsets_1d, shape):
        ndims = len(shape)
        offsets_long = []
        for o_1d in offsets_1d:
            o_nd = np.zeros(ndims, dtype=int)
            o_nd[axis] = o_1d
            o_long = to_long_index(o_nd, shape)
            offsets_long.append(o_long)
        return offsets_long


class _FinDiffUniformPeriodic(_FinDiffBase):

    def __init__(self, axis, order, spacing, acc):
        super().__init__(axis, order)
        self.spacing = spacing
        self.acc = acc
        self.coefs = coefficients(self.order, acc)["center"]

    def __call__(self, f):
        f = self.guard_valid_target(f)
        xp = get_namespace(f)

        fd = xp.zeros_like(f)
        for off, coef in zip(self.coefs["offsets"], self.coefs["coefficients"]):
            fd = fd + coef * xp.roll(f, -off, axis=self.axis)
        h_inv = 1.0 / self.spacing**self.order
        return fd * h_inv

    def write_matrix_entries(self, mat, shape):
        Is = get_long_indices_for_all_grid_points_as_1d_array(shape)
        h_inv = 1 / self.spacing**self.order
        for o, c in zip(self.coefs["offsets"], self.coefs["coefficients"]):
            Is_off = self._get_offset_indices_long(o, shape)
            mat[Is, Is_off] = c * h_inv

    def _get_offset_indices_long(self, o, shape):
        ndims = len(shape)
        idxs_short = [np.arange(n) for n in shape]
        idxs_short[self.axis] = np.roll(np.arange(shape[self.axis]), -o)
        grid = np.meshgrid(*idxs_short, indexing="ij")
        index_tuples = np.stack(grid, axis=-1).reshape(-1, ndims)
        Is_off = np.ravel_multi_index(index_tuples.T, shape)
        return Is_off


class _FinDiffNonUniform(_FinDiffBase):
    def __init__(self, axis, order, coords, acc):
        super().__init__(axis, order)
        self.coords = coords
        self.acc = acc

        schemes = calc_coefs_non_uni_batched(order, acc, coords)
        self.num_bndry = schemes["num_bndry"]
        self.forward = schemes["forward"]
        self.backward = schemes["backward"]
        self.center = schemes["center"]

    def __call__(self, y):
        """The core function to take a partial derivative on a non-uniform grid"""
        y = self.guard_valid_target(y)
        xp = get_namespace(y)

        dim = self.axis
        N = len(self.coords)
        yd = xp.zeros_like(y)
        ndims = len(y.shape)

        yd = self._apply_nonuni_weights(
            yd, y, dim, ndims,
            self.center["coefficients"], self.center["offsets"],
            slice(self.num_bndry, N - self.num_bndry),
        )

        yd = self._apply_nonuni_weights(
            yd, y, dim, ndims,
            self.forward["coefficients"], self.forward["offsets"],
            slice(0, self.num_bndry),
        )

        yd = self._apply_nonuni_weights(
            yd, y, dim, ndims,
            self.backward["coefficients"], self.backward["offsets"],
            slice(N - self.num_bndry, N),
        )

        return yd

    def _apply_nonuni_weights(self, yd, y, dim, ndims, weights, offsets, ref_slice):
        """Apply per-point weights using vectorized slice operations.

        Returns the updated output array (required for immutable backends
        like JAX).
        """
        n_pts = ref_slice.stop - ref_slice.start
        if n_pts == 0:
            return yd

        ref_multi = [slice(None)] * ndims
        ref_multi[dim] = ref_slice

        for j_idx, off in enumerate(offsets):
            off_slice = slice(ref_slice.start + int(off), ref_slice.stop + int(off))
            off_multi = [slice(None)] * ndims
            off_multi[dim] = off_slice

            w = weights[:, j_idx]
            w_shape = [1] * ndims
            w_shape[dim] = n_pts
            yd = add_at(yd, tuple(ref_multi), w.reshape(w_shape) * y[tuple(off_multi)])
        return yd

    def write_matrix_entries(self, mat, shape):
        long_indices_nd = get_long_indices_for_all_grid_points_as_ndarray(shape)
        N = len(self.coords)
        ndims = len(shape)

        schemes = [
            (self.forward, slice(0, self.num_bndry)),
            (self.center, slice(self.num_bndry, N - self.num_bndry)),
            (self.backward, slice(N - self.num_bndry, N)),
        ]

        for scheme, axis_slice in schemes:
            n_axis_pts = axis_slice.stop - axis_slice.start
            if n_axis_pts == 0:
                continue

            src_multi = [slice(None)] * ndims
            src_multi[self.axis] = axis_slice
            Is = long_indices_nd[tuple(src_multi)].reshape(-1)

            selected_shape = list(shape)
            selected_shape[self.axis] = n_axis_pts

            for j_idx, o in enumerate(scheme["offsets"]):
                tgt_multi = [slice(None)] * ndims
                tgt_multi[self.axis] = slice(
                    axis_slice.start + int(o), axis_slice.stop + int(o)
                )
                Js = long_indices_nd[tuple(tgt_multi)].reshape(-1)

                w = scheme["coefficients"][:, j_idx]
                w_shape = [1] * ndims
                w_shape[self.axis] = n_axis_pts
                w_broadcast = np.broadcast_to(
                    w.reshape(w_shape), selected_shape
                ).reshape(-1)

                mat[Is, Js] = w_broadcast
