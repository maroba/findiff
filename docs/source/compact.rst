Compact Finite Differences
==========================

Compact (or implicit) finite differences achieve spectral-like resolution with
small stencils by coupling derivative approximations at neighboring grid points.
The classical reference is Lele (1992).


Background
----------

Standard ("explicit") finite differences express a derivative as a weighted sum
of function values:

.. math::
   f'_i = \sum_k c_k \, f_{i+k}

Compact finite differences generalize this by including derivative values on the
left-hand side:

.. math::
   \sum_k \alpha_k \, f'_{i+k} = \sum_k c_k \, f_{i+k}

The classic tridiagonal case uses :math:`\alpha_{-1} = \alpha_1 = 1/3` and five
function values. Despite needing only nearest neighbors, this scheme reaches
6th-order accuracy — the same as a 7-point explicit stencil.

The price is that computing the derivative requires solving a banded linear
system. For a tridiagonal left-hand side this costs :math:`O(N)` and is fast in
practice.


Usage
-----

There are two ways to set up a compact scheme.

**Explicit scheme definition** — you pick the left-hand side coefficients
(:math:`\alpha`) and right-hand side offsets, and findiff solves for the matching
:math:`c_k`::

    import numpy as np
    from findiff import Diff, CompactScheme

    x = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    dx = x[1] - x[0]

    # Lele (1992) tridiagonal scheme, Section 2.1.1
    scheme = CompactScheme(
        deriv=1,
        left={-1: 1/3, 0: 1, 1: 1/3},
        right=[-3, -2, -1, 0, 1, 2, 3],
    )

    d_dx = Diff(0, dx, scheme=scheme, periodic=True)

    f = np.sin(x)
    df = d_dx(f)            # 6th-order accurate

**Shortcut** — let findiff choose a scheme that meets a target accuracy::

    d_dx = Diff(0, dx, compact=3, acc=6, periodic=True)

The ``compact`` parameter sets the number of left-hand side points (must be an
odd integer). findiff will widen the right-hand side stencil until the requested
accuracy is reached.


Higher derivatives
------------------

Use the ``**`` operator::

    d2_dx2 = d_dx ** 2

This automatically constructs a new compact scheme for the second derivative
with matching accuracy. It works the same way as with regular finite differences.


Non-periodic boundaries
-----------------------

For non-periodic grids the interior stencil extends past the boundary at the
first and last few grid points. findiff handles this with one-sided compact
stencils (Visbal & Gaitonde, 2002): the left-hand side alphas and right-hand
side offsets are restricted to points that exist, and the system is re-solved for
the boundary rows. If the one-sided compact system turns out to be singular,
findiff falls back to standard explicit one-sided finite differences.

::

    x = np.linspace(0, 1, 100)
    dx = x[1] - x[0]

    d_dx = Diff(0, dx, scheme=scheme, periodic=False)

    f = np.sin(x)
    df = d_dx(f)            # accurate at interior *and* boundaries


Multi-dimensional grids
-----------------------

Compact differentiation extends to N-dimensional arrays via Kronecker products,
identically to how findiff handles regular finite differences::

    x = y = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    dx = x[1] - x[0]
    X, Y = np.meshgrid(x, y, indexing='ij')

    d_dx = Diff(0, dx, compact=3, acc=6, periodic=True)
    d_dy = Diff(1, dx, compact=3, acc=6, periodic=True)

    f = np.sin(X) * np.cos(Y)
    laplacian = (d_dx**2 + d_dy**2)(f)

You can freely mix compact and standard finite difference operators in the same
expression.


Matrix representation
---------------------

The matrix form of a compact operator is :math:`L^{-1} R`. You obtain it with
the usual ``matrix()`` call::

    M = d_dx.matrix((100,))
    df = M.dot(f)           # same result as d_dx(f)

Keep in mind that inverting the banded :math:`L` introduces fill-in, so the
matrix is denser than for explicit finite differences. For very large grids it
may be cheaper to apply the operator directly (which uses a sparse solve
internally) rather than forming the full matrix.


Limitations
-----------

- Only equidistant grids are supported. Passing a non-uniform grid raises
  ``NotImplementedError``.
- The ``stencil()`` method is not supported for compact schemes.


References
----------

- S. K. Lele, "Compact Finite Difference Schemes with Spectral-like Resolution",
  *J. Comp. Phys.* **103**, 16–42 (1992).

- M. R. Visbal and D. V. Gaitonde, "On the Use of Higher-Order Finite-Difference
  Schemes on Curvilinear and Deformed Meshes", *J. Comp. Phys.* **181**, 155–185
  (2002).
