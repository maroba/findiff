import numbers
from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse

from findiff.grids import EquidistantGrid, Grid, TensorProductGrid
from findiff.stencils import StencilSet


class Node(ABC):
    __array_priority__ = 100  # Makes sure custom multiplication is called over numpy's

    def __init__(self, *args, **kwargs):
        self.children = []

    @abstractmethod
    def __call__(self, f, *args, **kwargs):
        pass

    @abstractmethod
    def matrix(self, shape):
        pass

    def stencil(self, shape):
        return StencilSet(self, shape)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(self, other)

    def __sub__(self, other):
        return Add(ScalarOperator(-1) * other, self)

    def __rsub__(self, other):
        return Add(ScalarOperator(-1) * other, self)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(other, self)

    @property
    def grid(self):
        return getattr(self, "_grid", None)

    def set_grid(self, grid):
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

    DEFAULT_ACC = 2

    def __init__(self, axis=0, grid=None, acc=DEFAULT_ACC):
        super().__init__()

        if isinstance(grid, numbers.Number):
            grid = {axis: grid}
        elif hasattr(grid, "shape") and hasattr(grid, "__len__"):
            grid = TensorProductGrid({axis: grid})

        self.set_grid(grid)

        self.axis = axis
        self.order = 1
        self.acc = acc

        self._fd = None

    def __call__(self, f, *args, **kwargs):

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
        from findiff.legacy.operators import FinDiff

        if isinstance(self.grid, EquidistantGrid):
            spacing = self.grid.spacing[self.axis]
            self._fd = FinDiff(self.axis, spacing, self.order, acc=self.acc)
        elif isinstance(self.grid, TensorProductGrid):
            coords = self.grid.coords[self.axis]
            self._fd = FinDiff(self.axis, coords, self.order, acc=self.acc)

    def matrix(self, shape):
        if not self._fd:
            self._build_differentiator()
        return self._fd.matrix(shape)

    def __pow__(self, power):
        new_diff = Diff(self.axis, acc=self.acc, grid=self.grid)
        new_diff.order *= power
        return new_diff
