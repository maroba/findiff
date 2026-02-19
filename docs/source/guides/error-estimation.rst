================
Error Estimation
================

*findiff* can estimate the truncation error of a computed derivative by
comparing results at two accuracy orders. This helps you judge whether your
grid is fine enough and whether a higher accuracy order would be beneficial.


How It Works
------------

A finite difference scheme of accuracy order *p* has a truncation error that
scales as :math:`\mathcal{O}(h^p)`, where *h* is the grid spacing. By
computing the same derivative at two consecutive accuracy orders (*p* and
*p + 2*), the pointwise difference gives an estimate of the leading error
term:

.. math::

   f'_{\mathrm{acc}=p}   &= f'_{\mathrm{exact}} + C\, h^p + \mathcal{O}(h^{p+2}) \\
   f'_{\mathrm{acc}=p+2} &= f'_{\mathrm{exact}} + C'\, h^{p+2} + \mathcal{O}(h^{p+4})

.. math::

   \left| f'_{\mathrm{acc}=p} - f'_{\mathrm{acc}=p+2} \right| \approx |C|\, h^p

The higher-order result is itself a more accurate approximation and is
returned as the *extrapolated* value.


Basic Usage
-----------

.. code:: python

    import numpy as np
    from findiff import Diff

    x = np.linspace(0, 2 * np.pi, 200)
    dx = x[1] - x[0]
    f = np.sin(x)

    d_dx = Diff(0, dx)
    result = d_dx.estimate_error(f)

The return value is an ``ErrorEstimate`` named tuple with three fields:

.. code:: python

    result.derivative     # d/dx at acc=2 (the base accuracy)
    result.error          # pointwise absolute error estimate
    result.extrapolated   # d/dx at acc=4 (improved result)

You can also unpack it directly:

.. code:: python

    derivative, error, extrapolated = d_dx.estimate_error(f)


Choosing the Base Accuracy
--------------------------

By default the operator's current accuracy is used as the base. You can
override this with the ``acc`` parameter:

.. code:: python

    d_dx = Diff(0, dx)

    # Compare acc=4 vs acc=6
    result = d_dx.estimate_error(f, acc=4)


Composite Operators
-------------------

``estimate_error`` works on any differential operator expression, including
sums and products:

.. code:: python

    n = 100
    x = np.linspace(0, 2 * np.pi, n)
    y = np.linspace(0, 2 * np.pi, n)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing='ij')

    f = np.sin(X) * np.sin(Y)

    laplacian = Diff(0, dx)**2 + Diff(1, dy)**2
    result = laplacian.estimate_error(f)

    print("Max estimated error:", result.error.max())


Limitations
-----------

- **Compact schemes**: ``estimate_error`` is not supported when using compact
  (implicit) finite differences and will raise ``NotImplementedError``.
- **Boundary points**: Near the grid boundary, different stencils are used
  (forward/backward instead of central). The error estimate is still valid
  but may be less tight at these points.
