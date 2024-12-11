from findiff.legacy.diff import _Diff
from findiff.stencils import StencilSet

DEFAULT_ACC = 2


class _FinDiff:
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
       >>> X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
       >>> f = X**2 + Y**2 + Z**2

       To create :math:`\\frac{\\partial f}{\\partial x}` on a uniform grid with spacing dx, dy
       along the 0th axis or 1st axis, respectively, instantiate a FinDiff object and call it:

       >>> d_dx = _FinDiff(0, dx)
       >>> d_dy = _FinDiff(1, dx)
       >>> result = d_dx(f)

       For :math:`\\frac{\\partial^2 f}{\\partial x^2}` or :math:`\\frac{\\partial^2 f}{\\partial y^2}`:

       >>> d2_dx2 = _FinDiff(0, dx, 2)
       >>> d2_dy2 = _FinDiff(1, dy, 2)
       >>> result_2 = d2_dx2(f)
       >>> result_3 = d2_dy2(f)

       For :math:`\\frac{\\partial^4 f}{\partial x \\partial^2 y \\partial z}`, do:

       >>> op = _FinDiff((0, dx), (1, dy, 2), (2, dz))
       >>> result_4 = op(f)


    """

    def __init__(self, *args, **kwargs):
        self.acc = None
        self.spac = None
        self.pds = self._eval_args(args, kwargs)

    def __call__(self, rhs, *args, **kwargs):
        return self.apply(rhs, *args, **kwargs)

    def apply(self, rhs, *args, **kwargs):
        if "acc" not in kwargs:
            if self.acc is None:
                acc = DEFAULT_ACC
            else:
                acc = self.acc
            kwargs["acc"] = acc

        if len(args) == 0 and "h" not in kwargs:
            if self.uniform:
                args = (self.spac,)
            else:
                args = (self.coords,)
        return self.pds(rhs, *args, **kwargs)

    def stencil(self, shape):
        return StencilSet(self, shape)

    def matrix(self, shape, h=None, acc=None):
        if acc is None:
            if self.acc is None:
                acc = DEFAULT_ACC
            else:
                acc = self.acc

        if self.uniform:
            if h is None and self.spac is not None:
                h = self.spac
            return self.pds.matrix(shape, h=h, acc=acc)
        else:
            return self.pds.matrix(shape, coords=self.coords, acc=acc)

    def set_accuracy(self, acc):
        self.pds.set_accuracy(acc)

    def _eval_args(self, args, kwargs):
        spac = {}

        if "acc" in kwargs:
            self.acc = kwargs["acc"]

        if isinstance(args[0], tuple):  # mixed partial derivative
            pds = None

            for arg in args:
                if len(arg) == 3:
                    axis, h, order = arg
                elif len(arg) == 2:
                    axis, h = arg
                    order = 1
                else:
                    raise ValueError("Format: (axis, spacing, order=1)")
                spac[axis] = h
                if pds is None:
                    pds = _Diff(axis, order)
                else:
                    pd = _Diff(axis, order)
                    pds = pds * pd
        else:
            if len(args) == 3:
                axis, h, order = args
            elif len(args) == 2:
                axis, h = args
                order = 1
            else:
                raise ValueError("Format: (axis, spacing, order=1)")
            pds = _Diff(axis, order)

            spac[axis] = h

        # Check if spac is really the spacing and not the coordinates (nonuniform case)
        for a, s in spac.items():
            if hasattr(s, "__len__"):
                self.coords = spac
                self.uniform = False
                break
            else:
                self.spac = spac
                self.uniform = True
                break

        return pds
