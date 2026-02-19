==============================
Solving Partial Differential Equations
==============================

*findiff* can solve linear partial differential equations with Dirichlet,
Neumann or Robin boundary conditions using sparse matrix representations.

Setup
-----

.. code:: python

    import numpy as np
    from findiff import Diff, PDE, BoundaryConditions

The ``PDE`` class takes a differential operator and solves the resulting
linear system on a given grid.


Example: 1D Boundary Value Problem
-----------------------------------

Solve :math:`u''(x) = \sin(x)` on :math:`[0, \pi]` with :math:`u(0) = 0`
and :math:`u(\pi) = 0`:

.. code:: python

    x = np.linspace(0, np.pi, 100)
    dx = x[1] - x[0]

    # The differential equation: u'' = sin(x)
    L = Diff(0, dx)**2
    f = np.sin(x)

    # Boundary conditions
    bc = BoundaryConditions(x.shape)
    bc[0] = 0       # u(0) = 0      (Dirichlet)
    bc[-1] = 0      # u(pi) = 0     (Dirichlet)

    # Solve
    pde = PDE(L, f, bc)
    u = pde.solve()


Example: 2D Laplace Equation
------------------------------

Solve :math:`\nabla^2 u = 0` on a square domain:

.. code:: python

    x = y = np.linspace(0, 1, 50)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing='ij')

    L = Diff(0, dx)**2 + Diff(1, dy)**2
    f = np.zeros_like(X)

    bc = BoundaryConditions(X.shape)
    bc[0, :] = X[0, :]      # Dirichlet at x = 0
    bc[-1, :] = X[-1, :]    # Dirichlet at x = 1
    bc[:, 0] = Y[:, 0]      # Dirichlet at y = 0
    bc[:, -1] = Y[:, -1]    # Dirichlet at y = 1

    pde = PDE(L, f, bc)
    u = pde.solve()


Neumann Boundary Conditions
-----------------------------

For Neumann BCs, pass a ``Diff`` operator as the second element:

.. code:: python

    bc = BoundaryConditions(x.shape)
    bc[0] = 0                         # u(0) = 0
    bc[-1] = Diff(0, dx), 1.0         # u'(L) = 1.0


Robin (Mixed) Boundary Conditions
-----------------------------------

Robin boundary conditions have the form
:math:`\alpha \, u + \beta \, \frac{\partial u}{\partial n} = g`.
Specify them as a 4-tuple ``(alpha, diff_op, beta, g)``:

.. code:: python

    bc = BoundaryConditions(x.shape)
    bc[0] = 1                                       # Dirichlet: u(0) = 1
    bc[-1] = (1, Diff(0, dx), 1, 3)                 # Robin: u + u' = 3

For 2D problems, Robin BCs on an edge work the same way:

.. code:: python

    bc = BoundaryConditions(X.shape)
    bc[-1, :] = (1, Diff(0, dx), 0.5, g_boundary)   # alpha*u + beta*du/dx = g
    bc[0, :] = u_left                                # Dirichlet on other edges
    bc[:, 0] = u_bottom
    bc[:, -1] = u_top

Alternatively, you can construct the Robin operator yourself using
``Identity()`` and ``Diff`` and pass it as a 2-tuple (like a Neumann BC):

.. code:: python

    robin_op = alpha * Identity() + beta * Diff(0, dx)
    bc[-1] = robin_op, g


Eigenvalue Problems
--------------------

Differential operators can solve eigenvalue problems of the form
:math:`L\,u = \lambda\,u` using the ``eigsh`` method (for symmetric
operators) or ``eigs`` (for general operators). These are thin wrappers
around ``scipy.sparse.linalg.eigsh`` / ``eigs``.

**Example: vibration modes of a string**

The eigenvalue problem :math:`u'' = \lambda\,u` on :math:`[0, \pi]` with
:math:`u(0) = u(\pi) = 0` has exact eigenvalues :math:`\lambda_n = -n^2`:

.. code:: python

    x = np.linspace(0, np.pi, 200)
    dx = x[1] - x[0]
    L = Diff(0, dx)**2

    bc = BoundaryConditions(x.shape)
    bc[0] = 0
    bc[-1] = 0

    eigenvalues, eigenvectors = L.eigsh(x.shape, k=5, which='SM', bc=bc)
    # eigenvalues ≈ [-1, -4, -9, -16, -25]

Pass ``bc`` to eliminate boundary degrees of freedom (homogeneous
Dirichlet). Eigenvectors are returned with shape ``(*grid_shape, k)``
— access the *i*-th mode as ``eigenvectors[..., i]``.

For generalized eigenvalue problems :math:`L\,u = \lambda\,M\,u`, pass a
second operator via the ``M`` parameter:

.. code:: python

    eigenvalues, eigenvectors = L.eigsh(shape, k=5, which='SM', bc=bc, M=M_op)

See the :doc:`matrix-representation` guide for a more detailed example
with the Schrodinger equation.


Iterative Solvers
------------------

By default, ``PDE.solve()`` uses the direct solver
``scipy.sparse.linalg.spsolve``.  For large problems (especially in 3D),
iterative solvers can be more memory-efficient and faster.  Pass the
``solver`` keyword to select an iterative method:

.. code:: python

    u = pde.solve(solver='gmres')

The following solver names are supported:

.. list-table::
   :header-rows: 1
   :widths: 15 40 45

   * - Name
     - Method
     - When to use
   * - ``'direct'``
     - ``scipy.sparse.linalg.spsolve``
     - Default. Reliable for small to medium problems.
   * - ``'cg'``
     - Conjugate Gradient
     - Symmetric positive definite systems only.
   * - ``'gmres'``
     - Generalized Minimal Residual
     - General (non-symmetric) systems. Good default iterative choice.
   * - ``'bicgstab'``
     - BiConjugate Gradient Stabilized
     - General non-symmetric systems.
   * - ``'lgmres'``
     - LGMRES
     - Variant of GMRES with improved convergence in some cases.
   * - ``'minres'``
     - Minimal Residual
     - Symmetric indefinite systems.

**Solver options** such as tolerance, maximum iterations, and initial
guess are passed as additional keyword arguments:

.. code:: python

    u = pde.solve(solver='gmres', tol=1e-10, maxiter=1000)

    # Provide an initial guess close to the expected solution
    u = pde.solve(solver='bicgstab', x0=u_previous)

**Preconditioners** can dramatically improve convergence.  Use the
``preconditioner='ilu'`` shorthand for an incomplete LU factorization:

.. code:: python

    u = pde.solve(solver='gmres', preconditioner='ilu')

Or pass a custom ``scipy.sparse.linalg.LinearOperator``:

.. code:: python

    from scipy.sparse.linalg import LinearOperator

    M = LinearOperator(...)  # your custom preconditioner
    u = pde.solve(solver='gmres', preconditioner=M)

**Custom solver callables** are also supported.  Any function with
signature ``f(A, b, **kw) -> x`` or ``f(A, b, **kw) -> (x, info)``
can be used:

.. code:: python

    from scipy.sparse.linalg import spsolve
    u = pde.solve(solver=lambda A, b: spsolve(A, b))

If an iterative solver fails to converge, a ``RuntimeError`` is raised.
