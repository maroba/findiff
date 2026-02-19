=================
Polar Coordinates
=================

By assembling general linear combinations of differential operators with
variable coefficients, you can use vector calculus operators in coordinates
other than Cartesian. Here we demonstrate polar coordinates in 2D.

.. code:: python

    import numpy as np
    from findiff import Diff, Laplacian


Setup: Cartesian Laplacian
--------------------------

Consider the 2D paraboloid :math:`f(x, y) = x^2 + y^2`, whose Laplacian is
trivially :math:`\nabla^2 f = 4` everywhere.

.. code:: python

    x, y = [np.linspace(-5, 5, 100)] * 2
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing='ij')
    f = X**2 + Y**2

    laplace = Laplacian(h=[dx, dy])
    laplace_f = laplace(f)  # array of 4.0 everywhere


Polar coordinates
-----------------

In polar coordinates the same function is :math:`f(r, \varphi) = r^2`, and
the Laplacian is:

.. math::

   \nabla^2 = \frac{\partial^2}{\partial r^2}
   + \frac{1}{r}\frac{\partial}{\partial r}
   + \frac{1}{r^2}\frac{\partial^2}{\partial \varphi^2}

In *findiff*:

.. code:: python

    r = np.linspace(0.1, 10, 100)
    phi = np.linspace(0, 2*np.pi, 100, endpoint=False)
    dr, dphi = r[1] - r[0], phi[1] - phi[0]
    R, Phi = np.meshgrid(r, phi, indexing='ij')
    f_polar = R**2

    laplace_polar = (
        Diff(0, dr)**2
        + (1/R) * Diff(0, dr)
        + (1/R**2) * Diff(1, dphi)**2
    )
    result = laplace_polar(f_polar)  # array of 4.0 everywhere
