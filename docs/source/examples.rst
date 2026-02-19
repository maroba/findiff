========
Examples
========

Jupyter notebooks demonstrating *findiff* in action. These notebooks are
in the ``examples/`` directory of the
`repository <https://github.com/maroba/findiff/tree/master/examples>`_.


Basic Usage
-----------

`examples-basic.ipynb <https://github.com/maroba/findiff/blob/master/examples/examples-basic.ipynb>`_

First and higher-order derivatives on 1D and multi-dimensional grids,
general differential operators with constant and variable coefficients,
and accuracy control.


Non-Uniform Grids
-----------------

`examples-non-uniform-grids.ipynb <https://github.com/maroba/findiff/blob/master/examples/examples-non-uniform-grids.ipynb>`_

Computing derivatives on grids with non-equidistant spacing, useful when
higher resolution is needed in regions of high curvature.


Periodic Boundary Conditions
-----------------------------

`examples-periodic.ipynb <https://github.com/maroba/findiff/blob/master/examples/examples-periodic.ipynb>`_

Derivatives on periodic domains. Demonstrates how wrapping at boundaries
produces circulant matrix representations.


Polar Coordinates
-----------------

`examples-polar.ipynb <https://github.com/maroba/findiff/blob/master/examples/examples-polar.ipynb>`_

Using *findiff* in non-Cartesian coordinate systems by constructing the
Laplacian in polar coordinates from its component derivatives.


Stencils
--------

`examples-stencils.ipynb <https://github.com/maroba/findiff/blob/master/examples/examples-stencils.ipynb>`_

Inspecting automatic stencils and creating custom stencils for
non-standard grid patterns (e.g. diagonal neighbors for the 2D Laplacian).


Vector Calculus
---------------

`examples-vector-calculus.ipynb <https://github.com/maroba/findiff/blob/master/examples/examples-vector-calculus.ipynb>`_

Gradient, divergence, curl and Laplacian using the convenience classes
on 3D grids.


Symbolic Representation
-----------------------

`symbolic.ipynb <https://github.com/maroba/findiff/blob/master/examples/symbolic.ipynb>`_

Using ``SymbolicMesh`` and ``SymbolicDiff`` to derive finite difference
schemes symbolically with *sympy*.
