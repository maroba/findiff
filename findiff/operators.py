from .stencils import Stencil
from .utils import *
from .diff import Coef, Id, Diff


def FinDiff(*args, **kwargs):
    """ Mocks the deprecated FinDiff class by wrapping the new Diff class """

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
            if pds is None:
                pds = Diff(axis, order)
                pds.h = h
            else:
                pd = Diff(axis, order)
                pd.h = h
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
        pds.h = h

    return pds


# Alias for backward compatibility
Coefficient = Coef
Identity = Id

