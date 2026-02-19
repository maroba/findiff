===============
Vector Calculus
===============

*findiff* implements the standard vector calculus operations

.. math::

   \nabla\;, \quad \nabla \cdot\;, \quad \nabla^2\;, \quad \nabla \times

via the convenience classes ``Gradient``, ``Divergence``, ``Laplacian`` and
``Curl``, respectively.

Setup
-----

.. code:: python

    import numpy as np
    from findiff import Gradient, Divergence, Laplacian, Curl

    x, y, z = [np.linspace(0, 10, 100)] * 3
    dx, dy, dz = [c[1] - c[0] for c in (x, y, z)]
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    f = np.sin(X) * np.cos(Y) * np.sin(Z)


Gradient
--------

The gradient of a scalar field yields a vector field:

.. code:: python

    grad = Gradient(h=[dx, dy, dz])
    grad_f = grad(f)
    grad_f.shape  # (3, 100, 100, 100)


Laplacian
---------

The Laplacian of a scalar field yields a scalar field:

.. code:: python

    laplace = Laplacian(h=[dx, dy, dz])
    laplace_f = laplace(f)
    laplace_f.shape  # (100, 100, 100)


Divergence
----------

Define a vector field and compute its divergence:

.. code:: python

    g = np.array([f, 2*f, 3*f])  # shape: (3, 100, 100, 100)

    div = Divergence(h=[dx, dy, dz])
    div_g = div(g)
    div_g.shape  # (100, 100, 100)


Curl
----

The curl of a vector field yields another vector field:

.. code:: python

    curl = Curl(h=[dx, dy, dz])
    curl_g = curl(g)
    curl_g.shape  # (3, 100, 100, 100)

.. note::
   The curl is only defined for three dimensions. Defining the operator on
   some other dimension raises an exception.
