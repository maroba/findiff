================
Basic Derivatives
================

*findiff* works in any number of dimensions. Here we demonstrate the basics
with 1D and 3D examples on uniform (equidistant) grids.

Setup
-----

.. code:: python

    import numpy as np
    from findiff import Diff, coefficients


Simple 1D Case
--------------

Suppose we want to compute the second derivative of
:math:`f(x) = \sin(x)` and :math:`g(x) = \cos(x)`:

.. math::

   \frac{d^2f}{dx^2} = -\sin(x) \quad \mbox{and}\quad \frac{d^2g}{dx^2} = -\cos(x)

.. code:: python

    x = np.linspace(0, 10, 100)
    dx = x[1] - x[0]
    f = np.sin(x)
    g = np.cos(x)

Construct the differential operator :math:`\frac{d^2}{dx^2}`:

.. code:: python

    d2_dx2 = Diff(0, dx) ** 2

Apply it:

.. code:: python

    result_f = d2_dx2(f)
    result_g = d2_dx2(g)

That's it! The result arrays have the same shape as the input and contain
the values of the second derivatives.


Finite Difference Coefficients
------------------------------

By default ``Diff`` uses second order accuracy. You can inspect the
finite difference coefficients:

.. code:: python

    coefficients(deriv=2, acc=2)

.. code:: python

    {'backward': {'coefficients': array([-1.,  4., -5.,  2.]),
      'offsets': array([-3, -2, -1,  0])},
     'center': {'coefficients': array([ 1., -2.,  1.]),
      'offsets': array([-1,  0,  1])},
     'forward': {'coefficients': array([ 2., -5.,  4., -1.]),
      'offsets': array([0, 1, 2, 3])}}

Higher accuracy orders are easy:

.. code:: python

    coefficients(deriv=2, acc=10)
    d2_dx2 = Diff(0, dx, acc=10) ** 2


Simple 3D Case
--------------

Differentiate :math:`f(x, y, z) = \sin(x) \cos(y) \sin(z)`:

.. code:: python

    x, y, z = [np.linspace(0, 10, 100)] * 3
    dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    f = np.sin(X) * np.cos(Y) * np.sin(Z)

Partial derivatives:

.. code:: python

    d_dx = Diff(0, dx)
    d_dz = Diff(2, dz)

Mixed partial derivative :math:`\frac{\partial^2 f}{\partial x \partial y}`:

.. code:: python

    d2_dxdy = Diff(0, dx) * Diff(1, dy)
    result = d2_dxdy(f)


General Linear Differential Operators
--------------------------------------

``Diff`` objects can be added and multiplied by numbers or arrays:

.. code:: python

    # Constant coefficients
    linear_op = Diff(0, dx)**2 + 2 * Diff(0, dx) * Diff(1, dy) + Diff(1, dy)**2

    # Variable coefficients
    linear_op = X * Diff(0, dx) + Y**2 * Diff(1, dy)

    result = linear_op(f)
