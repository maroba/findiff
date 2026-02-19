===========
**findiff**
===========

A Python package for finite difference numerical derivatives
and partial differential equations in any number of dimensions.

.. image:: images/findiff_logo.png
    :width: 300
    :align: center


Features
--------

- Differentiate arrays of any number of dimensions along any axis with any desired accuracy order
- Accurate treatment of grid boundary
- Includes standard operators from vector calculus like gradient, divergence and curl
- Can handle uniform and non-uniform grids
- Can handle arbitrary linear combinations of derivatives with constant and variable coefficients
- Fully vectorized for speed
- Calculate raw finite difference coefficients for any order and accuracy
- Generate matrix representations of arbitrary linear differential operators
- Solve partial differential equations with Dirichlet, Neumann or Robin boundary conditions
- Solve eigenvalue problems (e.g. Schrodinger equation, vibration modes)
- Generate differential operators for arbitrary stencils
- Symbolic representation of finite difference schemes
- Periodic boundary conditions for differential operators and PDEs
- Compact (implicit) finite differences with spectral-like resolution
- Estimate truncation error by comparing accuracy orders

.. versionadded:: 0.11
   Comfortable new API via ``Diff`` (the old ``FinDiff`` API remains available)

.. versionadded:: 0.12
   Periodic boundary conditions for differential operators and PDEs

.. versionadded:: 0.13
   Compact (implicit) finite differences with spectral-like resolution

.. versionadded:: 0.14
   Error estimation via accuracy order comparison


Content
-------

.. toctree::
    :maxdepth: 2

    getting-started/index
    guides/index
    theory/index
    api/index
    citation


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
