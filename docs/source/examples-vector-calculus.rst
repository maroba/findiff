
Vector Calculus
===============

*findiff* implements the standard vector calculus operations

.. math::


   \left(
   \frac{\partial}{\partial x_0},
   \frac{\partial}{\partial x_1},
   \dots,
   \frac{\partial}{\partial x_{N-1}}
   \right)\;, \quad
   \nabla \cdot\;, \quad
   \nabla^2\;, \quad
   \nabla \times

by the convenience classes ``Gradient``, ``Divergence``, ``Laplace`` and
``Curl``, respectively.

.. code:: ipython3

    import numpy as np
    from findiff import Gradient, Divergence, Laplacian, Curl

First, we want to apply the gradient, the divergence and the Laplacian
to some scalar function

.. math::


   f(x, y, z) = \sin(x) \cos(y) \sin(z)

We set up our grid and fill the array ``f``:

.. code:: ipython3

    x, y, z = [np.linspace(0, 10, 100)] * 3
    dx, dy, dz = [c[1] - c[0] for c in (x, y, z)]
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    f = np.sin(X) * np.cos(Y) * np.sin(Z)

:math:`f(x, y, z)` is a function of three variables, so the array ``f``
has three axes:

.. code:: ipython3

    f.shape




.. parsed-literal::

    (100, 100, 100)



It is a scalar function, so we can apply the gradient:

.. code:: ipython3

    grad = Gradient(h=[dx, dy, dz])
    grad_f = grad(f)

Applying the gradient yields a vector function, where each component is
a function of three variables. So the shape of the gradient array is:

.. code:: ipython3

    grad_f.shape




.. parsed-literal::

    (3, 100, 100, 100)



Applying the Laplacian to a scalar function yields another scalar
function:

.. code:: ipython3

    laplace = Laplacian(h=[dx, dy, dz])
    laplace_f = laplace(f)
    laplace_f.shape




.. parsed-literal::

    (100, 100, 100)



Now we define a vector function

.. math::


   {\bf g}(x, y, z) = \left(
   f(x, y, z), 2\cdot f(x, y, z), 3\cdot f(x, y, z)
   \right)

.. code:: ipython3

    g = np.array([f, 2*f, 3*f])
    g.shape




.. parsed-literal::

    (3, 100, 100, 100)



Applying the divergence yields a scalar function:

.. code:: ipython3

    div = Divergence(h=[dx, dy, dz])
    div_g = div(g)
    div_g.shape




.. parsed-literal::

    (100, 100, 100)



Applying the curl yields another vector function:

.. code:: ipython3

    curl = Curl(h=[dx, dy, dz])
    curl_g = curl(g)
    curl_g.shape




.. parsed-literal::

    (3, 100, 100, 100)



Note that the curl is only defined for three dimensions. Defining the
operator on some other dimension raises an exception.
