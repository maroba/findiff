import numbers
from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse

from findiff.grids import EquidistantGrid, Grid, TensorProductGrid
from findiff.stencils import StencilSet


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

    def __init__(self, axis=0, grid=None, acc=DEFAULT_ACC):
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
        acc: (optional) int
            The accuracy order to use. Must be a positive even number.
        """
        super().__init__()

        if isinstance(grid, numbers.Number):
            grid = {axis: grid}
        elif hasattr(grid, "shape") and hasattr(grid, "__len__"):
            grid = TensorProductGrid({axis: grid})

        self.set_grid(grid)

        self.axis = axis
        self._order = 1
        self.acc = acc

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
            self._fd = _FinDiff(self.axis, spacing, self.order, acc=self.acc)
        elif isinstance(self.grid, TensorProductGrid):
            coords = self.grid.coords[self.axis]
            self._fd = _FinDiff(self.axis, coords, self.order, acc=self.acc)

    def matrix(self, shape):
        if not self._fd:
            self._build_differentiator()
        return self._fd.matrix(shape)

    def __pow__(self, power):
        """Returns a Diff instance for a higher order derivative."""
        new_diff = Diff(self.axis, acc=self.acc, grid=self.grid)
        new_diff._order *= power
        return new_diff
