============
PDE Cookbook
============

A collection of worked examples solving classical partial differential
equations with *findiff*. Each recipe is self-contained.

.. contents:: Recipes
   :local:


1D Forced Harmonic Oscillator
------------------------------

Solve

.. math::

   u'' - \alpha\, u' + \omega^2\, u = \cos(2t)

on :math:`[0, 10]` with :math:`u(0)=0,\; u(10)=1`.

.. code:: python

    import numpy as np
    from findiff import Diff, Identity, PDE, BoundaryConditions

    N = 300
    t = np.linspace(0, 10, N)
    dt = t[1] - t[0]

    alpha = 1.0
    omega = np.sqrt(5)

    L = Diff(0, dt)**2 - alpha * Diff(0, dt) + omega**2 * Identity()
    f = np.cos(2 * t)

    bc = BoundaryConditions((N,))
    bc[0] = 0
    bc[-1] = 1

    pde = PDE(L, f, bc)
    u = pde.solve()


2D Laplace Equation (Steady-State Heat)
-----------------------------------------

Solve :math:`\nabla^2 u = 0` on a unit square with temperature fixed on
all four edges:

.. code:: python

    import numpy as np
    from findiff import Diff, PDE, BoundaryConditions

    N = 80
    x = y = np.linspace(0, 1, N)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing='ij')

    L = Diff(0, dx)**2 + Diff(1, dy)**2
    f = np.zeros_like(X)

    bc = BoundaryConditions(X.shape)
    bc[0, :] = 0                   # left edge: u = 0
    bc[-1, :] = 0                  # right edge: u = 0
    bc[:, 0] = 0                   # bottom edge: u = 0
    bc[:, -1] = np.sin(np.pi * x)  # top edge: u = sin(pi*x)

    pde = PDE(L, f, bc)
    u = pde.solve()


2D Poisson Equation
--------------------

Solve :math:`\nabla^2 u = -2\pi^2 \sin(\pi x)\sin(\pi y)` on a unit
square with :math:`u=0` on all boundaries (exact solution:
:math:`u = \sin(\pi x)\sin(\pi y)`).

.. code:: python

    import numpy as np
    from findiff import Diff, PDE, BoundaryConditions

    N = 80
    x = y = np.linspace(0, 1, N)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing='ij')

    L = Diff(0, dx)**2 + Diff(1, dy)**2
    f = -2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)

    bc = BoundaryConditions(X.shape)
    bc[0, :] = 0
    bc[-1, :] = 0
    bc[:, 0] = 0
    bc[:, -1] = 0

    pde = PDE(L, f, bc)
    u = pde.solve()

    # Check
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    print("Max error:", np.max(np.abs(u - u_exact)))


2D Heat Conduction with Mixed BCs
------------------------------------

A plate with a temperature profile on one edge, zero heat flux across
the others:

.. code:: python

    import numpy as np
    from findiff import Diff, PDE, BoundaryConditions

    N = 100
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing='ij')

    L = Diff(0, dx)**2 + Diff(1, dy)**2
    f = np.zeros_like(X)

    bc = BoundaryConditions(X.shape)
    bc[1, :] = Diff(0, dx), 0          # Neumann: du/dx = 0 at x = 0
    bc[-1, :] = 300. - 200 * Y         # Dirichlet at x = 1
    bc[:, 0] = 300.                     # Dirichlet at y = 0
    bc[1:-1, -1] = Diff(1, dy), 0      # Neumann: du/dy = 0 at y = 1

    pde = PDE(L, f, bc)
    u = pde.solve()


1D Schrodinger Eigenvalue Problem
-----------------------------------

Find the lowest energy levels of the quantum harmonic oscillator
:math:`-\tfrac{1}{2}\psi'' + \tfrac{1}{2}x^2\psi = E\psi`:

.. code:: python

    import numpy as np
    from findiff import Diff, Identity, BoundaryConditions

    N = 300
    x = np.linspace(-8, 8, N)
    dx = x[1] - x[0]

    H = -0.5 * Diff(0, dx)**2 + 0.5 * x**2 * Identity()

    bc = BoundaryConditions((N,))
    bc[0] = 0
    bc[-1] = 0

    eigenvalues, eigenvectors = H.eigsh((N,), k=6, which='SM', bc=bc)
    print("Eigenvalues:", eigenvalues)
    # Expected: ≈ [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]


Advection-Diffusion with Robin BC
------------------------------------

Solve

.. math::

   D\, u'' - v\, u' = 0

on :math:`[0, 1]` with :math:`u(0)=1` and a Robin condition
:math:`u + \tfrac{D}{v} u' = 0` at :math:`x=1` (outflow condition):

.. code:: python

    import numpy as np
    from findiff import Diff, PDE, BoundaryConditions

    N = 200
    x = np.linspace(0, 1, N)
    dx = x[1] - x[0]

    D = 0.01   # diffusion coefficient
    v = 1.0    # advection velocity

    L = D * Diff(0, dx)**2 - v * Diff(0, dx)
    f = np.zeros(N)

    bc = BoundaryConditions((N,))
    bc[0] = 1                                   # Dirichlet: u(0) = 1
    bc[-1] = (1, Diff(0, dx), D / v, 0)         # Robin: u + (D/v)*u' = 0

    pde = PDE(L, f, bc)
    u = pde.solve()


Large 2D Problem with Iterative Solver
-----------------------------------------

For large grids, iterative solvers with preconditioning can be faster and
use less memory than the default direct solver:

.. code:: python

    import numpy as np
    from findiff import Diff, PDE, BoundaryConditions

    N = 300
    x = y = np.linspace(0, 1, N)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing='ij')

    L = Diff(0, dx)**2 + Diff(1, dy)**2
    f = np.ones_like(X)

    bc = BoundaryConditions(X.shape)
    bc[0, :] = 0
    bc[-1, :] = 0
    bc[:, 0] = 0
    bc[:, -1] = 0

    pde = PDE(L, f, bc)
    u = pde.solve(solver='cg', preconditioner='ilu', tol=1e-10)
