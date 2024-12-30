"""This module provides an interface to obsolete classes for backward compatibility."""

from findiff.interface import Diff
from findiff.operators import FieldOperator, Identity


def FinDiff(*args, **kwargs):
    r"""A representation of a general linear differential operator expressed in finite differences.

        FinDiff objects can be added with other FinDiff objects. They can be multiplied by
        objects of type Coefficient.

        FinDiff is callable, i.e. to apply the derivative, just call the object on the array to
        differentiate.

        :param args: variable number of tuples. Defines what derivative to take.
            If only one tuple is given, you can leave away the tuple parentheses.

        Each tuple has the form

               `(axis, spacing, count)`     for uniform grids

               `(axis, count)`              for non-uniform grids.

             `axis` is the dimension along which to take derivative.

             `spacing` is the grid spacing of the uniform grid along that axis.

             `count` is the order of the derivative, which is optional an defaults to 1.


        :param kwargs:  variable number of keyword arguments

            Allowed keywords:

            `acc`:    even integer
                  The desired accuracy order. Default is acc=2.

        This class is actually deprecated and will be replaced by the Diff class in the future.

    **Example**:


       For this example, we want to operate on some 3D array f:

       >>> import numpy as np
       >>> x, y, z = [np.linspace(-1, 1, 100) for _ in range(3)]
       >>> dx = x[1] - x[0]
       >>> dy = y[1] - y[0]
       >>> dz = z[1] - z[0]
       >>> X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
       >>> f = X**2 + Y**2 + Z**2

       To create :math:`\\frac{\\partial f}{\\partial x}` on a uniform grid with spacing dx, dy
       along the 0th axis or 1st axis, respectively, instantiate a FinDiff object and call it:

       >>> d_dx = FinDiff(0, dx)
       >>> d_dy = FinDiff(1, dy)
       >>> result = d_dx(f)

       For :math:`\\frac{\\partial^2 f}{\\partial x^2}` or :math:`\\frac{\\partial^2 f}{\\partial y^2}`:

       >>> d2_dx2 = FinDiff(0, dx, 2)
       >>> d2_dy2 = FinDiff(1, dy, 2)
       >>> result_2 = d2_dx2(f)
       >>> result_3 = d2_dy2(f)

       For :math:`\\frac{\\partial^4 f}{\partial x \\partial^2 y \\partial z}`, do:

       >>> op = FinDiff((0, dx), (1, dy, 2), (2, dz))
       >>> result_4 = op(f)


    """
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


###
### Define aliasses for backward compatibility:
###


class Coefficient(FieldOperator):
    pass


Coef = Coefficient
Id = Identity
