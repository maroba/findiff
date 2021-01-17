===============
Getting Started
===============

Installation
::::::::::::

.. code-block:: ipython

    pip install --upgrade findiff

First Derivatives
:::::::::::::::::

The first derivative along the 0-th axis ("x0-axis"),

.. math::

    \frac{\partial}{\partial x_0}\quad,

can be defined by

.. code-block:: ipython

    from findiff import FinDiff

    d_dx = FinDiff(0, dx)

The first argument is the **axis** along which to take the partial derivative.
The second argument is the **spacing** of the (equidistant) grid along that axis.

Accordingly, the first partial derivative with respect to the `k`-th axis

.. math:: \frac{\partial}{\partial x_k}

is

.. code-block:: ipython

    FinDiff(k, dx_k)

Let's initialize a one-dimensional array ``f`` with some values, for example:

.. code-block:: ipython

    import numpy as np
    x = np.linspace(-np.pi, np.pi, 100)
    dx = x[1] - x[0]
    f = np.sin(x)

and calculate the first derivative with respect of the zeroth axis.

``FinDiff`` objects behave like operators, so in order to apply them, you can
simply call them on a *numpy* ``ndarray`` of any shape:


.. code-block:: ipython

    d_dx = FinDiff(0, dx)
    df_dx = d_dx(f)

Now ``df_dx`` is a new `numpy` array with the same shape as ``f`` containing the
first derivative with respect to the zeroth axis:

.. image:: scripts/get_started_plot_1.png


Higher Derivatives
::::::::::::::::::

The `n`-th partial derivatives, say with respect to :math:`x_k`,

.. math:: \frac{\partial^n}{\partial x_k^n}

is

.. code-block:: ipython

    FinDiff(k, dx_k, n)

where the last argument stands for the degree of the derivative.

A **mixed partial derivatives** like

.. math:: \frac{\partial^3}{\partial x^2 \partial y}

is defined by

.. code-block:: ipython

    FinDiff((0, dx, 2), (1, dy, 1))

where for each partial derivative, there is a tuple of the form
``(axis, spacing, degree)`` in the argument list.


General Differential Operators
::::::::::::::::::::::::::::::

``FinDiff`` objects can be combined to describe general differential
operators. For example, the wave operator

.. math::

    \frac{1}{c^2}\frac{\partial^2}{\partial t^2} - \frac{\partial^2}{\partial x^2}

can be written as

.. code-block:: ipython

    1 / c**2 * FinDiff(0, dt, 2) - FinDiff(1, dx, 2)

if the 0-axis represents the `t`-axis and the 1-axis the `x`-axis.

Non-constant coefficients must be wrapped as ``Coef`` objects. For instance,

.. math:: x^2 \frac{\partial^2}{\partial x^2}

is written as

.. code-block:: ipython

    x = np.linspace(-1, 1, 21)
    Coef(x) * FinDiff(0, dx, 2)

Finally, multiplication of two ``FinDiff`` objects means chaining differential
operators, for example

.. math::

    \left(\frac{\partial}{\partial x} - \frac{\partial}{\partial y}\right) \cdot
    \left(\frac{\partial}{\partial x} + \frac{\partial}{\partial y}\right)
    = \frac{\partial^2}{\partial x^2} - \frac{\partial^2}{\partial y^2}

or in `findiff`:

.. code-block:: ipython

    d_dx = FinDiff(0, dx, 1)
    d_dy = FinDiff(1, dx, 1)

    (d_dx - d_dy) * (d_dx + d_dy)


Accuracy Control
::::::::::::::::

By default, `findiff` uses finite difference schemes with
second order accuracy in the grid spacing. Higher orders can be selected
by setting the keyword argument ``acc``, e.g.

.. code-block:: ipython

    FinDiff(0, dx, 2, acc=4)

for fourth order accuracy.


Finite Difference Coefficients
::::::::::::::::::::::::::::::

`findiff` uses finite difference schemes to calculate numerical derivatives.
If needed, the finite difference coefficients can be obtained from the
``coefficients`` function, e.g. for second derivative with second order
accuracy:

.. code-block:: ipython

    from findiff import coefficients
    coefficients(deriv=2, acc=2)

which yields

.. code-block:: ipython

       {'backward': {'accuracy': 2,
                     'coefficients': array([-1.,  4., -5.,  2.]),
                     'offsets': array([-3, -2, -1,  0])},
        'center': {'accuracy': 2,
                   'coefficients': array([ 1., -2.,  1.]),
                   'offsets': array([-1,  0,  1])},
        'forward': {'accuracy': 2,
                    'coefficients': array([ 2., -5.,  4., -1.]),
                    'offsets': array([0, 1, 2, 3])}}

Matrix Representation
:::::::::::::::::::::

For a given FinDiff differential operator, you can get the matrix
representation using the matrix(shape) method, e.g.

.. code-block:: ipython

    x = [np.linspace(0, 6, 7)]
    d2_dx2 = FinDiff(0, x[1]-x[0], 2)
    u = x**2

    mat = d2_dx2.matrix(u.shape)  # this method returns a scipy sparse matrix
    print(mat.toarray())

yields

.. code-block:: ipython

    [[ 2. -5.  4. -1.  0.  0.  0.]
     [ 1. -2.  1.  0.  0.  0.  0.]
     [ 0.  1. -2.  1.  0.  0.  0.]
     [ 0.  0.  1. -2.  1.  0.  0.]
     [ 0.  0.  0.  1. -2.  1.  0.]
     [ 0.  0.  0.  0.  1. -2.  1.]
     [ 0.  0.  0. -1.  4. -5.  2.]]

Of course this also works for general differential operators.

