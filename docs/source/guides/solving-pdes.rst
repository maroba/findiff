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
