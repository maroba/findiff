"""Provides an interface to obsolete classes for backward compatibility."""

from findiff import Diff
from findiff.legacy.operators import _FinDiff
from findiff.operators import FieldOperator, Identity


def FinDiff(*args, **kwargs):

    if len(args) > 3:
        raise ValueError("FinDiff accepts not more than 3 positional arguments.")

    def diff_from_tuple(tpl):
        if len(tpl) == 3:
            axis, h, order = tpl
            return Diff(axis, h, **kwargs) ** order
        elif len(tpl) == 2:
            axis, h = tpl
            return Diff(axis, h, **kwargs)

    if isinstance(args[0], (list, tuple)):
        diffs = []
        for tpl in args:
            diffs.append(diff_from_tuple(tpl))
        fd = diffs[0]
        for diff in diffs[1:]:
            fd = fd * diff
        return fd

    return diff_from_tuple(args)


FinDiff.__doc__ = _FinDiff.__doc__


# Define aliasses for backward compatibility
Coefficient = Coef = FieldOperator
Id = Identity
