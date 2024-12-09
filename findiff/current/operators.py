import numbers
from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse


class Node(ABC):
    __array_priority__ = 100  # Make sure custom multiplication is called over numpy's

    @abstractmethod
    def __call__(self, f, *args, **kwargs):
        pass

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(self, other)

    def __sub__(self, other):
        return Add(FieldOperator(-1) * other, self)

    def __rsub__(self, other):
        return Add(FieldOperator(-1) * other, self)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(other, self)


class FieldOperator(Node):
    """An operator that multiplies an array pointwise."""

    def __init__(self, value):
        self.value = value

    def __call__(self, f, *args, **kwargs):
        if isinstance(f, (numbers.Number, np.ndarray)):
            return self.value * f
        return self.value * super().__call__(f, *args, **kwargs)


class ScalarOperator(FieldOperator):
    def __init__(self, value):
        if not isinstance(value, numbers.Number):
            raise ValueError("Expected number, got " + str(type(value)))
        super().__init__(value)


class Identity(ScalarOperator):
    def __init__(self):
        super().__init__(1)


class Add(Node):
    def __init__(self, left, right):
        if isinstance(left, (numbers.Number, np.ndarray)):
            left = FieldOperator(left)
        if isinstance(right, (numbers.Number, np.ndarray)):
            right = FieldOperator(right)
        self.left = left
        self.right = right

    def __call__(self, f, *args, **kwargs):
        return self.left(f, *args, **kwargs) + self.right(f, *args, **kwargs)


class Mul(Node):
    def __init__(self, left, right):
        if isinstance(left, (numbers.Number, np.ndarray)):
            left = FieldOperator(left)
        if isinstance(right, (numbers.Number, np.ndarray)):
            right = FieldOperator(right)
        self.left = left
        self.right = right

    def __call__(self, f, *args, **kwargs):
        return self.left(self.right(f, *args, **kwargs), *args, **kwargs)


class Diff(Node):

    DEFAULT_ACC = 2

    def __init__(self, spacing, order=1, axis=0, acc=DEFAULT_ACC):
        if spacing <= 0:
            raise ValueError("spacing must be greater than zero")
        if order <= 0:
            raise ValueError("order must be greater than zero")

        self.spacing = spacing
        self.axis = axis
        self.order = order
        self.acc = acc

        self._fd = None

    def __call__(self, f, *args, **kwargs):

        if "acc" in kwargs:
            # allow to pass down new accuracy deep in expression tree
            new_acc = kwargs["acc"]

            if new_acc != self.acc:
                self._fd = None

            self.acc = new_acc

        if self._fd is None:
            self._build_differentiator()

        if isinstance(f, Node):
            f = f(*args, **kwargs)
        return self._fd(f)

    def _build_differentiator(self):
        from findiff.legacy import FinDiff

        self._fd = FinDiff(self.axis, self.spacing, self.order, acc=self.acc)
