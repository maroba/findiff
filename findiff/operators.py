import numbers
from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse

from findiff.findiff import build_differentiator
from findiff.grids import GridAxis, make_grid
from findiff.stencils import StencilSet


class Expression(ABC):
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
        self._grid = make_grid(grid)
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


class FieldOperator(Expression):
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


class BinaryOperation(Expression):

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


class Diff(Expression):

    DEFAULT_ACC = 2

    def __init__(self, dim, axis: GridAxis = None, acc=DEFAULT_ACC):
        """Initializes a Diff instance.

        Parameters
        ----------
        dim: int
            The 0-based index of the axis along which to take the derivative.
        axis: GridAxis
            The grid axis.
        acc: (optional) int
            The accuracy order to use. Must be a positive even number.
        """
        super().__init__()

        self.set_axis(axis)
        self.dim = dim
        self.acc = acc
        self._order = 1
        self._differentiator = None

    def set_grid(self, grid):
        super().set_grid(grid)
        self.set_axis(self.grid.get_axis(self.dim))

    def set_axis(self, axis: GridAxis):
        self._axis = axis
        self._differentiator = None

    @property
    def axis(self):
        return self._axis

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
                self._differentiator = None
                self.set_accuracy(new_acc)

        if isinstance(f, Expression):
            f = f(*args, **kwargs)

        return self.differentiator(f)

    @property
    def differentiator(self):
        if self._differentiator is None:
            self._differentiator = build_differentiator(self.order, self.axis, self.acc)
        return self._differentiator

    def matrix(self, shape):
        return self.differentiator.matrix(shape)

    def __pow__(self, power):
        """Returns a Diff instance for a higher order derivative."""
        new_diff = Diff(self.dim, self.axis, acc=self.acc)
        new_diff._order *= power
        return new_diff

    def __mul__(self, other):
        if isinstance(other, Diff) and self.dim == other.dim:
            new_diff = Diff(self.dim, self.axis, acc=self.acc)
            new_diff._order += other.order
            return new_diff
        return super().__mul__(other)
