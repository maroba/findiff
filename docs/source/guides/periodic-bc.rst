==========================
Periodic Boundary Conditions
==========================

.. versionadded:: 0.12

*findiff* supports periodic boundary conditions for differential operators
and PDEs. This is useful for problems on periodic domains, such as
Fourier-like settings.

Setup
-----

.. code:: python

    import numpy as np
    from findiff import Diff

    # Periodic grid: endpoint=False is important!
    x = np.linspace(0, 2*np.pi, 100, endpoint=False)
    dx = x[1] - x[0]
    f = np.sin(x)


Periodic derivatives
--------------------

Pass ``periodic=True`` to enable periodic boundary conditions:

.. code:: python

    d_dx = Diff(0, dx, periodic=True)
    df = d_dx(f)

The derivative wraps around at the boundaries — the last grid point
is treated as the neighbor of the first.


Higher derivatives
------------------

Works the same way:

.. code:: python

    d2_dx2 = Diff(0, dx, periodic=True) ** 2
    d2f = d2_dx2(f)


Multi-dimensional periodic grids
---------------------------------

Periodicity can be enabled per axis:

.. code:: python

    x = np.linspace(0, 2*np.pi, 50, endpoint=False)
    y = np.linspace(0, 2*np.pi, 50, endpoint=False)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing='ij')
    f = np.sin(X) * np.cos(Y)

    # Periodic in both x and y
    d_dx = Diff(0, dx, periodic=True)
    d_dy = Diff(1, dy, periodic=True)
    laplacian = d_dx**2 + d_dy**2
    result = laplacian(f)


Matrix representation
---------------------

The matrix representation of periodic operators uses a circulant structure:

.. code:: python

    mat = d_dx.matrix(f.shape)
