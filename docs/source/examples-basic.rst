
Basic Examples of *findiff*
===========================

*findiff* works in any dimension. But for the sake of demonstration,
let's concentrate on the cases 1D and 3D. We are using uniform, i.e.
equidistant, grids here. The non-uniform case will be shown in another
notebook.

Preliminaries
-------------

Our imports:

.. code:: ipython3

    import numpy as np
    from findiff import Diff, coefficients

Simple 1D Cases
---------------

Suppose we want to differentiate two 1D-arrays ``f`` and ``g``, which
are filled with values from a function

.. math::


   f(x) = \sin(x) \quad \mbox{and}\quad g(x) = \cos(x)

and we want to take the 2nd derivative. This is easy done analytically:

.. math::


   \frac{d^2f}{dx^2} = -\sin(x) \quad \mbox{and}\quad \frac{d^2g}{dx^2} = -\cos(x)

Let's do this numerically with *findiff*. First we set up the grid and
the arrays:

.. code:: ipython3

    x = np.linspace(0, 10, 100)
    dx = x[1] - x[0]
    f = np.sin(x)
    g = np.cos(x)

Then we construct the derivative object, which represents the
differential operator :math:`\frac{d^2}{dx^2}`:

.. code:: ipython3

    d_dx = Diff(0, dx)

The first parameter is the axis along which to take the derivative.
Since we want to apply it to the one and only axis of the 1D array, this
is a 0. The second parameter describes the grid to be used. In our case,
we have an equidistant grid point, so a single number (the grid spacing along
the desired axis) suffices. `Diff` always returns a first derivative. If
you need higher order derivatives, use exponentiation:

.. code:: ipython3

    d2_dx2 = Diff(0, dx) ** 2

Then we apply the operator to f and g, respectively:

.. code:: ipython3

    result_f = d2_dx2(f)
    result_g = d2_dx2(g)

That's it! The arrays ``result_f``\ and ``result_g`` have the same shape
as the arrays ``f`` and ``g`` and contain the values of the second
derivatives.

Finite Difference Coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default the ``Diff`` class uses second order accuracy. For the
second derivative, it uses the following finite difference coefficients:

.. code:: ipython3

    coefficients(deriv=2, acc=2)


.. parsed-literal::

    {'backward': {'coefficients': array([-1.,  4., -5.,  2.]),
      'offsets': array([-3, -2, -1,  0])},
     'center': {'coefficients': array([ 1., -2.,  1.]),
      'offsets': array([-1,  0,  1])},
     'forward': {'coefficients': array([ 2., -5.,  4., -1.]),
      'offsets': array([0, 1, 2, 3])}}


But ``Diff`` can handle any accuracy order. For instance, have you
ever wondered, what the 10th order accurate coefficients look like? Here
they are:

.. code:: ipython3

    coefficients(deriv=2, acc=10)




.. parsed-literal::

    {'backward': {'coefficients': array([  -0.53253968,    6.42373016,  -35.55158728,  119.41369042,
             -271.26190464,  439.39444427, -521.11333314,  457.02976176,
             -295.51984119,  138.59325394,  -44.43730158,    7.56162698]),
      'offsets': array([-11, -10,  -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,   0])},
     'center': {'coefficients': array([ 3.17460317e-04, -4.96031746e-03,  3.96825397e-02, -2.38095238e-01,
              1.66666667e+00, -2.92722222e+00,  1.66666667e+00, -2.38095238e-01,
              3.96825397e-02, -4.96031746e-03,  3.17460317e-04]),
      'offsets': array([-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5])},
     'forward': {'coefficients': array([   7.56162876,  -44.43731776,  138.59331976, -295.52000468,
              457.03003946, -521.1136706 ,  439.39474213, -271.26209495,
              119.41377646,  -35.55161345,    6.42373497,   -0.53254009]),
      'offsets': array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])}}



Accuracy order
^^^^^^^^^^^^^^

If you want to use for example 10th order accuracy, just tell the
``Diff`` constructor to use it:

.. code:: ipython3

    d2_dx2 = Diff(0, dx, acc=10) ** 2
    result = d2_dx2(f)

Simple 3D Cases
---------------

Now let's differentiate a 3D-array ``f`` representing the function

.. math::


   f(x, y, z) = \sin(x) \cos(y) \sin(z) 

.. code:: ipython3

    x, y, z = [np.linspace(0, 10, 100)]*3
    dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    f = np.sin(X) * np.cos(Y) * np.sin(Z)

The partial derivatives :math:`\frac{\partial f}{\partial x}` or
:math:`\frac{\partial f}{\partial z}` are given by

.. code:: ipython3

    d_dx = Diff(0, dx)
    d_dz = Diff(2, dz)

The x-axis is the 0th axis, y, the first, z the 2nd, etc. The
mixed partial derivative
:math:`\frac{\partial^2 f}{\partial x \partial y}` is specified by multiplying
the two first order partial derivatives:

.. code:: ipython3

    d2_dxdy = Diff(0, dx) * Diff(1, dy)
    result = d2_dxdy(f)

Of course, the accuracy order can be specified the same way as for 1D.

General Linear Differential Operators
-------------------------------------

``Diff`` objects can bei added and easily multiplied by numbers. For
example, to express

.. math::


   \frac{\partial^2}{\partial x^2} + 2\frac{\partial^2}{\partial x \partial y} + \frac{\partial^2}{\partial y^2} =
   \left(\frac{\partial}{\partial x} + \frac{\partial}{\partial y}\right) \left(\frac{\partial}{\partial x} + \frac{\partial}{\partial y}\right)

we can say

.. code:: ipython3

    linear_op = Diff(0, dx)**2 + 2 * Diff(0, dx) * Diff(1, dy) + Diff(1, dy)**2

If you want to multiply by variables instead of plain numbers, it works the same way.
For example,

.. math::


   x \frac{\partial}{\partial x} + y^2 \frac{\partial}{\partial y}

is

.. code:: ipython3

    linear_op = X * Diff(0, dx) + Y**2 * Diff(1, dy)

Applying those general operators works the same way as for the simple
derivatives:

.. code:: ipython3

    result = linear_op(f)
