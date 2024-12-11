
Polar Coordinates
=================

By assembling general linear combinations of differential operators with
variable coefficients in *findiff*, you can use vector calculus
operators in coordinates other than cartesian. Here we show an example
for using polar coordinates in 2D.

.. code:: ipython3

    import numpy as np
    from findiff import Diff, Laplacian

Sample Problem
--------------

In this example, we calculate the Laplacian of the 2D paraboloid

.. math::


   f(x, y) = x^2 + y^2

once in cartesian coordinates and once in polar coordinates. Let's start
with cartesian.

The Laplacian is obviously

.. math::


   \nabla^2 f = 2 + 2 = 4

for all points :math:`(x, y)`.

Now let's do this with *findiff*. First we define the grid and the
function.

.. code:: ipython3

    x, y = [np.linspace(-5, 5, 100)] * 2
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing='ij')
    f = X**2 + Y**2

Then we define and apply the Laplacian in cartesian coordinates:

.. code:: ipython3

    laplace = Laplacian(h=[dx, dy])
    laplace_f = laplace(f)

We get

.. code:: ipython3

    laplace_f




.. parsed-literal::

    array([[4., 4., 4., ..., 4., 4., 4.],
           [4., 4., 4., ..., 4., 4., 4.],
           [4., 4., 4., ..., 4., 4., 4.],
           ...,
           [4., 4., 4., ..., 4., 4., 4.],
           [4., 4., 4., ..., 4., 4., 4.],
           [4., 4., 4., ..., 4., 4., 4.]])



Polar Coordinates
~~~~~~~~~~~~~~~~~

In polar coordinates the paraboloid is simply

.. math::


   f(r, \varphi) = r^2

We define our grid in polar coordinates and calculate the :math:`f` in
polar coordinates:

.. code:: ipython3

    r = np.linspace(0.1, 10, 100)
    phi = np.linspace(0, 2*np.pi, 100, endpoint=False)
    dr, dphi = r[1] - r[0], phi[1] - phi[0]
    R, Phi = np.meshgrid(r, phi, indexing='ij')
    f_polar = R**2

Of course, the two arrays ``f`` and ``f_polar`` are completely
different:

.. code:: ipython3

    f - f_polar




.. parsed-literal::

    array([[ 49.99      ,  48.99010203,  48.01061014, ...,  48.01061014,
             48.99010203,  49.99      ],
           [ 48.96010203,  47.96020406,  46.98071217, ...,  46.98071217,
             47.96020406,  48.96010203],
           [ 47.93061014,  46.93071217,  45.95122028, ...,  45.95122028,
             46.93071217,  47.93061014],
           ...,
           [-48.01938986, -49.01928783, -49.99877972, ..., -49.99877972,
            -49.01928783, -48.01938986],
           [-49.00989797, -50.00979594, -50.98928783, ..., -50.98928783,
            -50.00979594, -49.00989797],
           [-50.        , -50.99989797, -51.97938986, ..., -51.97938986,
            -50.99989797, -50.        ]])



The Laplacian in polar coordinates is

.. math::


   \nabla^2 = \frac{\partial^2}{\partial r^2} + \frac{1}{r}\frac{\partial}{\partial r} + \frac{1}{r^2}\frac{\partial^2}{\partial \varphi^2}

or in *findiff*:

.. code:: ipython3

    laplace_polar = Diff(0, dr)**2 + (1/R) * Diff(0, dr) + (1/R**2) * Diff(1, dphi)**2
    result = laplace_polar(f_polar)

And we get the same result

.. code:: ipython3

    result




.. parsed-literal::

    array([[4., 4., 4., ..., 4., 4., 4.],
           [4., 4., 4., ..., 4., 4., 4.],
           [4., 4., 4., ..., 4., 4., 4.],
           ...,
           [4., 4., 4., ..., 4., 4., 4.],
           [4., 4., 4., ..., 4., 4., 4.],
           [4., 4., 4., ..., 4., 4., 4.]])


