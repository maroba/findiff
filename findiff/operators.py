from .stencils import Stencil
from .utils import *
from .diff import Coef, Id, Diff, Plus, Minus, Mul, DEFAULT_ACC, LinearMap


class FinDiff(LinearMap):
    """ Mocks the deprecated FinDiff class by wrapping the new Diff class """

    def __init__(self, *args, **kwargs):
        self.acc = None
        self.spac = None
        self.pds = self._eval_args(args, kwargs)

    def __call__(self, rhs, *args, **kwargs):
        return self.apply(rhs, *args, **kwargs)

    def apply(self, rhs, *args, **kwargs):
        if 'acc' not in kwargs:
            if self.acc is None:
                acc = DEFAULT_ACC
            else:
                acc = self.acc
            kwargs['acc'] = acc

        if len(args) == 0 and 'h' not in kwargs:
            if self.uniform:
                args = self.spac,
            else:
                args = self.coords,
        return self.pds(rhs, *args, **kwargs)

    def stencil(self, shape, h=None, acc=None, old_stl=None):
        if h is None and self.spac is not None:
            h = self.spac
        return self.pds.stencil(shape, h, acc, old_stl)

    def matrix(self, shape, h=None, acc=None):
        if h is None and self.spac is not None:
            h = self.spac
        if acc is None:
            acc = DEFAULT_ACC
        return self.pds.matrix(shape, h, acc)

    def set_accuracy(self, acc):
        self.pds.set_accuracy(acc)

    def __add__(self, other):
        return Plus(self, other)

    def __sub__(self, other):
        return Minus(self, other)

    def __mul__(self, other):
        return Mul(self, other)

    def _eval_args(self, args, kwargs):
        spac = {}

        if 'acc' in kwargs:
            self.acc = kwargs['acc']

        if isinstance(args[0], tuple): # mixed partial derivative
            pds = None

            for arg in args:
                if len(arg) == 3:
                    axis, h, order = arg
                elif len(arg) == 2:
                    axis, h = arg
                    order = 1
                else:
                    raise ValueError('Format: (axis, spacing, order=1)')
                spac[axis] = h
                if pds is None:
                    pds = Diff(axis, order)
                else:
                    pd = Diff(axis, order)
                    pds = pds * pd
        else:
            if len(args) == 3:
                axis, h, order = args
            elif len(args) == 2:
                axis, h = args
                order = 1
            else:
                raise ValueError('Format: (axis, spacing, order=1)')
            pds = Diff(axis, order)

            spac[axis] = h

        # Check if spac is really the spacing and not the coordinates (nonuniform case)
        for a, s in spac.items():
            if hasattr(s, '__len__'):
                self.coords = spac
                self.uniform = False
                break
            else:
                self.spac = spac
                self.uniform = True
                break

        return pds


# Alias for backward compatibility
Coefficient = Coef
Identity = Id

