import math
from itertools import product

import numpy as np

from .utils import to_index_tuple, to_long_index


class StencilSet:
    """
    Represent the finite difference stencil for a given differential operator.
    """

    def __init__(self, diff_op, shape, old_stl=None):
        """
        Constructor for Stencil objects.

        :param shape: tuple of ints
            Shape of the grid on which the stencil should be applied.

        :param axis: int >= 0
            The coordinate axis along which to take the partial derivative.

        :param order: int > 0
            The order of the derivative.

        :param h: float
            The spacing of the (equidistant) grid

        """

        self.shape = shape
        self.diff_op = diff_op
        self.char_pts = self._det_characteristic_points()

        self.data = {}

        self._create_stencil()

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return str(self.data)

    def apply(self, u, idx0):
        """ Applies the stencil to a point in an equidistant grid.

        :param u: ndarray
            An array with the function to differentiate.

        :param idx0: int or tuple of ints
            The index of the grid point where to differentiate the function.

        :return:
            The derivative at the given point.
        """

        if not hasattr(idx0, '__len__'):
            idx0 = (idx0,)

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
        """ Applies the stencil to all grid points.

        :param u: ndarray
            An array with the function to differentiate.

        :return:
            An array with the derivative.
        """

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

    def _create_stencil(self):

        matrix = self.diff_op.matrix(self.shape)

        for pt in self.char_pts:

            char_point_stencil = {}
            self.data[pt] = char_point_stencil

            index_tuple_for_char_pt = self._typical_index_tuple_for_char_point(pt)
            long_index_for_char_pt = to_long_index(index_tuple_for_char_pt, self.shape)

            row = matrix[long_index_for_char_pt, :]
            long_row_inds, long_col_inds = row.nonzero()

            for long_offset_ind in long_col_inds:
                offset_ind_tuple = np.array(to_index_tuple(long_offset_ind, self.shape), dtype=int)
                offset_ind_tuple -= np.array(index_tuple_for_char_pt, dtype=int)
                char_point_stencil[tuple(offset_ind_tuple)] = row[0, long_offset_ind]

    def _typical_index_tuple_for_char_point(self, pt):
        index_tuple_for_char_pt = []
        for axis, key in enumerate(pt):
            if key == 'L':
                index_tuple_for_char_pt.append(0)
            elif key == 'C':
                index_tuple_for_char_pt.append(self.shape[axis] // 2)
            else:
                index_tuple_for_char_pt.append(self.shape[axis] - 1)
        return tuple(index_tuple_for_char_pt)

    def _det_characteristic_points(self):
        shape = self.shape
        ndim = len(shape)
        typ = [("L", "C", "H")] * ndim
        return product(*typ)


class Stencil:

    def __init__(self, offsets, partials, spacings=None):

        self.partials = partials
        self.max_order = 100
        if not hasattr(offsets[0], "__len__"):
            ndims = 1
            self.offsets = [(off,) for off in offsets]
        else:
            ndims = len(offsets[0])
            self.offsets = offsets

        if spacings is None:
            spacings = [1] * ndims
        elif not hasattr(spacings, "__len__"):
            spacings = [spacings] * ndims
        assert len(spacings) == ndims
        self.spacings = spacings
        self.ndims = ndims
        self.sol, self.sol_as_dict = self._make_stencil()

    def __call__(self, f, at=None, on=None):
        if at is not None and on is None:
            return self._apply_at_single_point(f, at)
        if at is None and on is not None:
            if isinstance(on[0], slice):
                return self._apply_on_multi_slice(f, on)
            else:
                return self._apply_on_mask(f, on)
        raise Exception('Cannot specify both *at* and *on* parameters.')

    def __str__(self):
        return str(self.values)

    def __repr__(self):
        return str(self.values)

    def _apply_on_mask(self, f, mask):
        result = np.zeros_like(f)
        for offset, coeff in self.values.items():
            offset_mask = self._make_offset_mask(mask, offset)
            result[mask] += coeff * f[offset_mask]
        return result

    def _apply_on_multi_slice(self, f, on):
        result = np.zeros_like(f)
        base_mslice = [self._canonic_slice(sl, f.shape[axis]) for axis, sl in enumerate(on)]
        for off, coeff in self.values.items():
            off_mslice = list(base_mslice)
            for axis, off_ in enumerate(off):
                start = base_mslice[axis].start + off_
                stop = base_mslice[axis].stop + off_
                off_mslice[axis] = slice(start, stop)
            result[tuple(base_mslice)] += coeff * f[tuple(off_mslice)]
        return result

    def _apply_at_single_point(self, f, at):
        result = 0.
        at = np.array(at)
        for off, coeff in self.values.items():
            off = np.array(off)
            eval_at = at + off
            if np.any(eval_at < 0) or not np.all(eval_at < f.shape):
                raise Exception('Cannot evaluate outside of grid.')
            result += coeff * f[tuple(eval_at)]
        return result

    def _make_offset_mask(self, mask, offset):
        offset_mask = np.full_like(mask, fill_value=False, dtype=bool)
        mslice_off = []
        mslice_base = []
        for off_ in offset:
            if off_ == 0:
                sl_off = slice(None, None)
                sl_base = slice(None, None)
            elif off_ > 0:
                sl_off = slice(off_, None)
                sl_base = slice(None, -off_)
            else:
                sl_off = slice(None, off_)
                sl_base = slice(-off_, None)
            mslice_off.append(sl_off)
            mslice_base.append(sl_base)
        offset_mask[tuple(mslice_base)] = mask[tuple(mslice_off)]
        return offset_mask

    def _canonic_slice(self, sl, length):
        start = sl.start
        if start is None:
            start = 0
        if start < 0:
            start = length + start
        stop = sl.stop
        if stop is None:
            stop = length
        if stop < 0:
            stop = length + stop
        return slice(start, stop)

    @property
    def values(self):
        return self.sol_as_dict

    @property
    def accuracy(self):
        return self._calc_accuracy()

    def _calc_accuracy(self):
        tol = 1.E-6
        deriv_order = 0
        for pows in self.partials.keys():
            order = sum(pows)
            if order > deriv_order:
                deriv_order = order
        for order in range(deriv_order, deriv_order + 10):
            terms = self._multinomial_powers(order)
            for term in terms:
                row = self._system_matrix_row(term)
                resid = np.sum(np.array(self.sol) * np.array(row))
                if abs(resid) > tol and term not in self.partials:
                    return order - deriv_order

    def _make_stencil(self):
        sys_matrix, taylor_terms = self._system_matrix()
        rhs = [0] * len(self.offsets)

        for i, term in enumerate(taylor_terms):
            if term in self.partials:
                weight = self.partials[term]
                multiplicity = np.prod([math.factorial(a) for a in term])
                vol = np.prod([self.spacings[j] ** term[j] for j in range(self.ndims)])
                rhs[i] = weight * multiplicity / vol

        sol = np.linalg.solve(sys_matrix, rhs)
        assert len(sol) == len(self.offsets)
        return sol, {off: coef for off, coef in zip(self.offsets, sol) if coef != 0}

    def _system_matrix(self):
        rows = []
        used_taylor_terms = []
        for order in range(self.max_order):
            taylor_terms = self._multinomial_powers(order)
            for term in taylor_terms:
                rows.append(self._system_matrix_row(term))
                used_taylor_terms.append(term)
                if not self._rows_are_linearly_independent(rows):
                    rows.pop()
                    used_taylor_terms.pop()
                if len(rows) == len(self.offsets):
                    return np.array(rows), used_taylor_terms
        raise Exception('Not enough terms. Try to increase max_order.')

    def _system_matrix_row(self, powers):
        row = []
        for a in self.offsets:
            value = 1
            for i, power in enumerate(powers):
                value *= a[i] ** power
            row.append(value)
        return row

    def _multinomial_powers(self, the_sum):
        """Returns all tuples of a given dimension that add up to the_sum."""
        all_combs = list(product(range(the_sum + 1), repeat=self.ndims))
        return list(filter(lambda tpl: sum(tpl) == the_sum, all_combs))

    def _rows_are_linearly_independent(self, matrix):
        """Checks the linear independence of the rows of a matrix."""
        matrix = np.array(matrix).astype(float)
        return np.linalg.matrix_rank(matrix) == len(matrix)
