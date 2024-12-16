import itertools
import numbers
from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse

from findiff.coefs import coefficients, coefficients_non_uni
from findiff.grids import EquidistantGrid, Grid, TensorProductGrid
from findiff.stencils import StencilSet
from findiff.utils import long_indices_as_ndarray, to_long_index


class Node(ABC):
    """Represents a differential operator expression."""

    __array_priority__ = 100  # Makes sure custom multiplication is called over numpy's

    def __init__(self, *args, **kwargs):
        self.children = []

    @abstractmethod
    def __call__(self, f, *args, **kwargs):
        """Applies the differential operator expression L to array f.

        Parameters
        ----------
        f : numpy.ndarray
            The arrow to differentiate.

        Returns
        -------
        numpy.ndarray
            Array of the same shape as f containing the derivative values.
        """
        pass

    @abstractmethod
    def matrix(self, shape):
        """Returns a matrix representation of the differential operator for a given grid shape."""
        pass

    def stencil(self, shape):
        """Returns a stencil representation of the differential operator for a given grid shape."""
        return StencilSet(self, shape)

    def __add__(self, other):
        """Allows to add differential operator expressions."""
        return Add(self, other)

    def __radd__(self, other):
        """Allows to add differential operator expressions."""
        return Add(self, other)

    def __sub__(self, other):
        """Allows to subtract differential operator expressions."""
        return Add(ScalarOperator(-1) * other, self)

    def __rsub__(self, other):
        """Allows to subtract differential operator expressions."""
        return Add(ScalarOperator(-1) * other, self)

    def __mul__(self, other):
        """Allows to multiply differential operator expressions."""
        return Mul(self, other)

    def __rmul__(self, other):
        """Allows to multiply differential operator expressions."""
        return Mul(other, self)

    @property
    def grid(self):
        """Returns the grid used."""
        return getattr(self, "_grid", None)

    def set_grid(self, grid):
        """Sets the grid for the given differential operator expression.

        Parameters
        ----------
        grid: dict | Grid
            Specifies the grid to use. If a dict is given, an equidistant grid
            is assumed and the dict specifies the spacings along the required axes.
        """
        if isinstance(grid, dict):
            self._grid = EquidistantGrid(grid)
            for child in self.children:
                child.set_grid(self._grid)
        else:
            # Grid may be set lazily
            self._grid = grid
            for child in self.children:
                child.set_grid(self._grid)

    def set_accuracy(self, acc):
        """Sets the requested accuracy for the given differential operator expression.

        Parameters
        ----------
        acc: int
            The accuracy order. Must be a positive, even number.
        """
        self.acc = acc
        for child in self.children:
            child.set_accuracy(acc)


class FieldOperator(Node):
    """An operator that multiplies an array pointwise."""

    def __init__(self, value):
        super().__init__()
        self.value = value

    def __call__(self, f, *args, **kwargs):
        if isinstance(f, (numbers.Number, np.ndarray)):
            return self.value * f
        return self.value * super().__call__(f, *args, **kwargs)

    def matrix(self, shape):
        if isinstance(self.value, np.ndarray):
            diag_values = self.value.reshape(-1)
            return sparse.diags(diag_values)
        elif isinstance(self.value, numbers.Number):
            siz = np.prod(shape)
            return sparse.diags(self.value * np.ones(siz))


class ScalarOperator(FieldOperator):
    """A multiple of the identity operator."""

    def __init__(self, value):
        if not isinstance(value, numbers.Number):
            raise ValueError("Expected number, got " + str(type(value)))
        super().__init__(value)

    def matrix(self, shape):
        siz = np.prod(shape)
        mat = sparse.lil_matrix((siz, siz))
        diag = list(range(siz))
        mat[diag, diag] = self.value
        return sparse.csr_matrix(mat)


class Identity(ScalarOperator):
    """The identity operator."""

    def __init__(self):
        super().__init__(1)


class BinaryOperation(Node):

    @property
    def left(self):
        return self.children[0]

    @property
    def right(self):
        return self.children[1]


class Add(BinaryOperation):
    def __init__(self, left, right):
        if isinstance(left, (numbers.Number, np.ndarray)):
            left = FieldOperator(left)
        if isinstance(right, (numbers.Number, np.ndarray)):
            right = FieldOperator(right)
        super().__init__()
        self.children = [left, right]

    def __call__(self, f, *args, **kwargs):
        return self.left(f, *args, **kwargs) + self.right(f, *args, **kwargs)

    def matrix(self, shape):
        return self.left.matrix(shape) + self.right.matrix(shape)


class Mul(BinaryOperation):
    def __init__(self, left, right):
        if isinstance(left, (numbers.Number, np.ndarray)):
            left = FieldOperator(left)
        if isinstance(right, (numbers.Number, np.ndarray)):
            right = FieldOperator(right)
        super().__init__()
        self.children = [left, right]

    def __call__(self, f, *args, **kwargs):
        return self.left(self.right(f, *args, **kwargs), *args, **kwargs)

    def matrix(self, shape):
        return self.left.matrix(shape) * self.right.matrix(shape)


class Diff(Node):
    """Represents a partial derivative (along one axis).

    For higher derivatives, exponentiate. For mixed partial derivatives, multiply. See
    examples below.

    Examples
    --------
    Set up grid (equidistant here):
        >>> import numpy as np
        >>> x = np.linspace(0, 10, 100)

    The array to differentiate
        >>> f = np.sin(x) # as an example

    Define the first derivative:
        >>> from findiff import Diff
        >>> d_dx = Diff(0)
        >>> d_dx = d_dx.set_grid({0: x[1] - x[0]})

    Now apply it:
        >>> df_dx = d_dx(f)

    The second derivative is constructed by exponentiation:
        >>> d2_dx2 = d_dx ** 2
        >>> d2f_dx2 = d2_dx2(f)

    In multiple dimensions with meshed grids, the usage is accordingly:
        >>> x = y = z = np.linspace(0, 10, 100)
        >>> dx = dy = dz = x[1] - x[0]
        >>> X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        >>> f = np.sin(X) * np.sin(Y) * np.sin(Z)
        >>> d_dx = Diff(0)
        >>> d_dy = Diff(1)
        >>> d_dz = Diff(2)
        >>> d3_dxdydz = d_dx * d_dy * d_dz
        >>> d3_dxdydz.set_grid({0: dx, 1: dy, 2: dz})
        >>> d3f_dxdydz = d3_dxdydz(f)
    """

    DEFAULT_ACC = 2

    def __init__(self, axis=0, grid=None, periodic=False, acc=DEFAULT_ACC):
        """Initializes a Diff instance.

        Parameters
        ----------
        axis: int
            The 0-based index of the axis along which to take the derivative.
        grid: (optional) float | numpy.ndarray
            Specifies the grid to use. A float value assumes an equidistant grid
            and denoted the grid spacing along the given axis. A 1-D numpy array
            assumes an non-equidistant (tensor product) grid and denotes the coordinates
            along the given axis.
        periodic: bool
            Flag indicating where the axis is subject to periodic boundary conditions.
        acc: (optional) int
            The accuracy order to use. Must be a positive even number.
        """
        super().__init__()

        if isinstance(grid, numbers.Number):
            grid = {axis: {"h": grid, "periodic": periodic}}
        elif hasattr(grid, "shape") and hasattr(grid, "__len__"):
            grid = TensorProductGrid({axis: grid})

        self.set_grid(grid)

        self.axis = axis
        self._order = 1
        self.acc = acc

        self._fd = None

    def set_grid(self, grid):
        super().set_grid(grid)
        self._fd = None

    @property
    def order(self):
        """Returns the order of the derivative."""
        return self._order

    def __call__(self, f, *args, **kwargs):
        """Applies the differential operator."""

        if "acc" in kwargs:
            # allow to pass down new accuracy deep in expression tree
            new_acc = kwargs["acc"]
            if new_acc != self.acc:
                self._fd = None
                self.set_accuracy(new_acc)

        if self._fd is None:
            self._build_differentiator()

        if isinstance(f, Node):
            f = f(*args, **kwargs)
        return self._fd(f)

    def _build_differentiator(self):
        from findiff.legacy.operators import _FinDiff

        if isinstance(self.grid, EquidistantGrid):
            spacing = self.grid.spacing[self.axis]
            if not self.grid.periodic[self.axis]:
                self._fd = _FinDiffUniform(self.axis, self.order, spacing, self.acc)
            else:
                self._fd = _FinDiffUniformPeriodic(
                    self.axis, self.order, spacing, self.acc
                )
        elif isinstance(self.grid, TensorProductGrid):
            coords = self.grid.coords[self.axis]
            self._fd = _FinDiffNonUniform(self.axis, self.order, coords, self.acc)
        else:
            raise ValueError("Grid type corrupt.")

    def matrix(self, shape):
        if not self._fd:
            self._build_differentiator()
        return self._fd.matrix(shape)

    def __pow__(self, power):
        """Returns a Diff instance for a higher order derivative."""
        new_diff = Diff(self.axis, acc=self.acc, grid=self.grid)
        new_diff._order *= power
        return new_diff

    def __mul__(self, other):
        if isinstance(other, Diff) and self.axis == other.axis:
            new_diff = Diff(self.axis, acc=self.acc, grid=self.grid)
            new_diff._order += other.order
            return new_diff
        return super().__mul__(other)


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
