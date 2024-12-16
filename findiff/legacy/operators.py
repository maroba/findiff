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

       >>> d_dx = FinDiff(0, dx)
       >>> d_dy = FinDiff(1, dx)
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

    pass
