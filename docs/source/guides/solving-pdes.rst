==============================
Solving Partial Differential Equations
==============================

*findiff* can solve linear partial differential equations with Dirichlet
or Neumann boundary conditions using sparse matrix representations.

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
