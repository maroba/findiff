from copy import deepcopy
from findiff.operators import PartialDerivative, Plus, Operator, UnaryOperator, Multiply


class FinDiff(UnaryOperator):

    def __init__(self, *args, **kwargs):
        """
            Same args as PartialDerivative class. Describes a general linear differential operator.
        
        """

        self.acc = None
        if "acc" in kwargs:
            self.acc = kwargs["acc"]
            if self.acc % 2 == 1:
                self.acc += 1

        self.spac = None
        if "spac" in kwargs:
            self.spac = kwargs["spac"]

        self.coords = None
        if "coords" in kwargs:
            self.coords = kwargs["coords"]

        self.root = PartialDerivative(*args)
        self.child = None

    def __call__(self, u, **kwargs):

        for kwarg in kwargs:
            if kwarg == "spac":
                spac = kwargs[kwarg]
                if not hasattr(spac, "__getitem__"):
                    raise Exception("spac must be list or dict.")
                self.spac = spac
            elif kwarg == "acc":
                self.acc = kwargs[kwarg]
            elif kwarg == "coords":
                self.coords = kwargs[kwarg]
            else:
                raise Exception("Unknown kwarg.")

        if self.spac is None and self.coords is None:
            raise Exception("Neither grid spacing nor coordinates are set.")

        if self.acc is None:
            self.acc = 2

        if self.child is not None:
            u = self.child.apply(self, u)

        return self.root.apply(self, u)

    def is_uniform(self):
        if self.spac is not None and self.coords is not None:
            raise Exception("Both spac and coords set.")
        if self.spac:
            return True
        return False

    def __add__(self, other):
        fd = deepcopy(self)
        fd.root = Plus(fd.root, deepcopy(other))
        return fd

    def __rmul__(self, other):
        """
            'other' is the thing on the left side of '*'.        
        """

        if isinstance(other, Coef):
            mult = Multiply(other.value, deepcopy(self))
        else:
            mult = Multiply(other, deepcopy(self))
        fd = deepcopy(self)
        fd.root = mult

        return fd

    def __mul__(self, other):
        """Entered if self is FinDiff object in expression is self * other """

        if isinstance(other, Operator):
            self.child = other

        return self

    def apply(self, fd, u):
        return self.root.apply(fd, u)


class Coef(object):

    def __init__(self, value):
        self.value = value


class Identity(FinDiff):

    def __init__(self):
        super().__init__()
        self.spac = 0

