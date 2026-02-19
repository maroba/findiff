=================
Non-Uniform Grids
=================

*findiff* supports non-uniform (non-equidistant) grids. Instead of passing
a scalar grid spacing, pass the full coordinate array.

.. code:: python

    import numpy as np
    from findiff import Diff
    import matplotlib.pyplot as plt


Example function
----------------

Consider :math:`f(x) = x \, e^{-x^2}`:

.. code:: python

    def f(x):
        return x * np.exp(-x**2)

    def df_dx(x):
        return np.exp(-x**2) - 2*x**2*np.exp(-x**2)


Uniform grid limitations
------------------------

On a coarse uniform grid, the derivative is inaccurate where the function
has high curvature:

.. code:: python

    x_coarse = np.linspace(0, 10, 20)
    d_dx = Diff(0, x_coarse[1] - x_coarse[0])
    df = d_dx(f(x_coarse))  # large error near x < 3


Non-uniform grid
----------------

Instead, use a non-uniform grid with higher density where curvature is high:

.. code:: python

    x_nu = np.r_[
        np.linspace(0, 0.5, 3, endpoint=False),
        np.linspace(0.5, 1.2, 7, endpoint=False),
        np.linspace(1.2, 1.9, 2, endpoint=False),
        np.linspace(1.9, 2.9, 5, endpoint=False),
        np.linspace(2.9, 10, 3),
    ]

    # Pass the full coordinate array instead of a scalar spacing
    d_dx = Diff(0, x_nu, acc=2)
    df_nu = d_dx(f(x_nu))

The error is much smaller with the same number of points (20), because
the grid density matches the function's curvature.
