=====================
GPU / JAX / CuPy
=====================

*findiff* can transparently operate on **JAX** and **CuPy** arrays in addition
to standard NumPy arrays.  All derivative operators (``Diff``, ``Gradient``,
``Divergence``, ``Curl``, ``Laplacian``) and operator compositions (addition,
multiplication, exponentiation) work out of the box — just pass an array from
any supported backend.

Installation
============

*findiff* detects JAX and CuPy at runtime; neither is a hard dependency.
Install whichever backend you need:

.. code-block:: shell

    # JAX — CPU only
    pip install jax

    # JAX — NVIDIA GPU
    pip install jax[cuda12]

    # CuPy — NVIDIA GPU
    pip install cupy-cuda12x


Basic usage
===========

Pass a JAX (or CuPy) array where you would normally pass a NumPy array:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    from findiff import Diff

    jax.config.update("jax_enable_x64", True)

    x = jnp.linspace(0, 2 * jnp.pi, 1000)
    dx = float(x[1] - x[0])
    f = jnp.sin(x)

    d_dx = Diff(0, dx)
    df_dx = d_dx(f)          # returns a JAX array
    type(df_dx)               # jaxlib.xla_extension.ArrayImpl

The result stays on the same backend as the input — no implicit copies back
to NumPy.


Using ``jax.jit`` for speed
============================

Without JIT, JAX dispatches each NumPy-like operation individually, which can
be *slower* than plain NumPy due to dispatch overhead.  The real speedup comes
from **JIT compilation**, which fuses all the slice and arithmetic operations
inside the operator into a single optimized kernel:

.. code-block:: python

    d_dx_jit = jax.jit(d_dx)

    # First call traces + compiles (slow):
    result = d_dx_jit(f)

    # Subsequent calls reuse the compiled kernel (fast):
    result = d_dx_jit(f)

This works for any operator, including composed operators and vector calculus
shortcuts:

.. code-block:: python

    from findiff import Laplacian

    lap = Laplacian(h=[dx, dy, dz])
    lap_jit = jax.jit(lap)
    result = lap_jit(f_3d)

.. tip::

    Always call ``.block_until_ready()`` when benchmarking JAX, since JAX uses
    asynchronous dispatch::

        result = lap_jit(f_3d).block_until_ready()


Non-uniform grids
=================

Non-uniform grid coordinates can be passed as JAX or CuPy arrays.  They are
converted to NumPy internally for coefficient computation (which happens once
at operator construction time), while the operator application still runs on
the GPU backend:

.. code-block:: python

    import numpy as np

    x = np.linspace(0, np.pi, 500)        # NumPy coords for construction
    f = jnp.array(np.sin(x))              # JAX array for application

    d_dx = Diff(0, x)
    result = d_dx(f)                       # returns a JAX array


Vector calculus
===============

``Gradient``, ``Divergence``, ``Curl``, and ``Laplacian`` all support
alternative backends:

.. code-block:: python

    from findiff import Gradient, Laplacian

    grad = Gradient(h=[dx, dy], acc=4)
    grad_f = grad(f_2d_jax)               # returns a JAX array

    lap = Laplacian(h=[dx, dy])
    lap_f = lap(f_2d_jax)                 # returns a JAX array


Operator composition
====================

Composed operators with scalar or array coefficients work as expected:

.. code-block:: python

    from findiff import Diff, Identity

    d2 = Diff(0, dx) ** 2
    L = d2 + Identity()                    # d²/dx² + 1

    L_jit = jax.jit(L)
    result = L_jit(f_jax)


What is **not** supported on GPU
=================================

The following features remain **NumPy / SciPy only** and will raise errors or
return NumPy arrays if called with GPU data:

* ``.matrix(shape)`` — returns a ``scipy.sparse`` matrix
* ``PDE`` and ``BoundaryConditions`` — use ``scipy.sparse.linalg`` solvers
* ``TimeDependentPDE`` / ``MOLSolution`` — implicit time steppers use sparse solvers
* ``CompactScheme`` operators — use ``scipy.sparse.linalg.splu``
* ``Stencil`` / ``StencilSet`` — construction depends on the matrix path
* ``.eigs()`` / ``.eigsh()`` — use ``scipy.sparse.linalg``


Running the benchmarks
======================

The test suite includes benchmarks that compare NumPy and JAX performance.
They are excluded from normal ``pytest`` runs and can be executed with:

.. code-block:: shell

    pytest -m benchmark -v -s --override-ini='addopts='
