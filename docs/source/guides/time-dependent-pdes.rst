==============================
Time-Dependent PDEs
==============================

*findiff* can solve time-dependent PDEs of the form

.. math::

   \frac{\partial u}{\partial t} = L(u)

using the Method of Lines (MOL). The spatial operator :math:`L` is
discretised with *findiff*'s finite difference operators, and the
resulting system of ODEs is advanced in time with one of several
built-in time-stepping methods.

.. versionadded:: 0.14

.. contents:: Contents
   :local:


Quick Start: 1D Heat Equation
-------------------------------

Solve :math:`u_t = D\, u_{xx}` on :math:`[0, 1]` with
:math:`u(0)=u(1)=0` and initial condition :math:`u_0 = \sin(\pi x)`:

.. code:: python

    import numpy as np
    from findiff import Diff, TimeDependentPDE, BoundaryConditions

    nx = 101
    x = np.linspace(0, 1, nx)
    dx = x[1] - x[0]
    D = 0.01

    L = D * Diff(0, dx)**2
    u0 = np.sin(np.pi * x)

    bc = BoundaryConditions((nx,))
    bc[0] = 0
    bc[-1] = 0

    t = np.linspace(0, 1, 500)

    pde = TimeDependentPDE(L, u0, bc, t)
    u_final = pde.solve()   # returns the solution at t=1

    # Compare with exact solution
    u_exact = np.sin(np.pi * x) * np.exp(-D * np.pi**2 * t[-1])
    print("Max error:", np.max(np.abs(u_final - u_exact)))


Time-Stepping Methods
-----------------------

Four methods are available, selected via the ``method`` parameter of
:meth:`~findiff.ivp.TimeDependentPDE.solve`:

.. list-table::
   :header-rows: 1
   :widths: 25 15 20 40

   * - Method
     - Order
     - Type
     - Notes
   * - ``'forward-euler'``
     - 1
     - Explicit
     - Simple but requires small time steps (CFL condition).
   * - ``'rk4'``
     - 4
     - Explicit
     - High accuracy per step; still subject to CFL stability.
   * - ``'backward-euler'``
     - 1
     - Implicit
     - Unconditionally stable; low accuracy.
   * - ``'crank-nicolson'``
     - 2
     - Implicit
     - Unconditionally stable; good balance of accuracy and
       stability (default).

**Explicit methods** evaluate the spatial operator directly at each
step and are fast per step, but the time step is limited by stability
(roughly :math:`\Delta t < \Delta x^2 / (2D)` for diffusion problems).
They require all boundary conditions to be Dirichlet.

**Implicit methods** solve a sparse linear system at each step.  They
are unconditionally stable, so large time steps are possible.  They
support Dirichlet, Neumann, and Robin boundary conditions.

.. code:: python

    # Explicit: Forward Euler
    u = pde.solve(method='forward-euler')

    # Explicit: Runge-Kutta 4
    u = pde.solve(method='rk4')

    # Implicit: Backward Euler
    u = pde.solve(method='backward-euler')

    # Implicit: Crank-Nicolson (default)
    u = pde.solve(method='crank-nicolson')


2D Heat Equation
-----------------

Solve :math:`u_t = D\,(\nabla^2 u)` on a unit square with zero
Dirichlet boundary conditions:

.. code:: python

    import numpy as np
    from findiff import Diff, TimeDependentPDE, BoundaryConditions

    nx, ny = 51, 51
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing='ij')

    D = 0.01
    L = D * (Diff(0, dx)**2 + Diff(1, dy)**2)

    u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)

    bc = BoundaryConditions((nx, ny))
    bc[0, :] = 0
    bc[-1, :] = 0
    bc[:, 0] = 0
    bc[:, -1] = 0

    t = np.linspace(0, 0.5, 200)
    u_final = TimeDependentPDE(L, u0, bc, t).solve()


Advection with Periodic BCs
-----------------------------

Solve :math:`u_t = -c\, u_x` on :math:`[0, 2\pi)` with periodic
boundary conditions:

.. code:: python

    import numpy as np
    from findiff import Diff, TimeDependentPDE, BoundaryConditions

    nx = 201
    c = 1.0
    x = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    dx = x[1] - x[0]

    L = -c * Diff(0, dx, periodic=True)
    u0 = np.sin(x)

    # Empty BCs (periodic operator handles wrapping internally)
    bc = BoundaryConditions((nx,))

    t = np.linspace(0, 2 * np.pi, 1000)
    u_final = TimeDependentPDE(L, u0, bc, t).solve(method='rk4')

    # After one full period the wave should return to start
    print("Max error:", np.max(np.abs(u_final - u0)))


Time-Dependent Boundary Conditions and Sources
-----------------------------------------------

.. versionadded:: 0.16

Boundary values can be functions of time: assign a callable ``g(t)`` instead
of a number. This works for Dirichlet, Neumann and Robin conditions. The
callable may return a scalar, an array with the shape of the selected boundary
points, or an array with the shape of the whole grid.

A source term :math:`s` turns the problem into
:math:`u_t = L(u) + s(t)`. Pass it as an array (constant in time) or as a
callable ``s(t)``:

.. code-block:: python

    import numpy as np
    from findiff import Diff, TimeDependentPDE, BoundaryConditions

    nx = 101
    x = np.linspace(0, 1, nx)
    dx = x[1] - x[0]

    bc = BoundaryConditions((nx,))
    bc[0] = lambda t: np.sin(2 * np.pi * t)       # oscillating temperature
    bc[-1] = Diff(0, dx), lambda t: 0.1 * t       # growing heat flux

    heating = lambda t: np.exp(-((x - 0.5) / 0.1) ** 2) * (t < 1)

    t = np.linspace(0, 2, 401)
    u = TimeDependentPDE(0.05 * Diff(0, dx)**2, np.zeros(nx), bc, t,
                         source=heating).solve()

Boundary values and sources are evaluated at the times each method requires
(including the intermediate stages of RK4), so all methods keep their order of
accuracy. Explicit methods still require Dirichlet conditions only.

``PDE`` (for boundary value problems) does not accept time-dependent boundary
conditions.


Second Order in Time: the Wave Equation
---------------------------------------

.. versionadded:: 0.16

Pass the initial velocity ``v0`` to solve problems of second order in time,
:math:`u_{tt} = L(u) + s(t)`, such as the wave equation
:math:`u_{tt} = c^2 u_{xx}`:

.. code-block:: python

    import numpy as np
    from findiff import Diff, TimeDependentPDE, BoundaryConditions

    nx, c = 201, 1.0
    x = np.linspace(0, 1, nx)
    dx = x[1] - x[0]

    bc = BoundaryConditions((nx,))
    bc[0] = 0
    bc[-1] = 0

    u0 = np.exp(-((x - 0.5) / 0.05) ** 2)    # Gaussian pulse at rest
    t = np.arange(0, 2, 0.5 * dx / c)

    sol = TimeDependentPDE(c**2 * Diff(0, dx)**2, u0, bc, t,
                           v0=np.zeros(nx)).solve(method="verlet", store_every=10)
    sol.u    # displacement snapshots
    sol.v    # velocity snapshots

Available methods:

- ``'verlet'``: explicit Stoermer-Verlet, second order. Conserves energy over
  long times; the method of choice for waves. Stable for
  :math:`c\,\Delta t / \Delta x \lesssim 1` (Dirichlet conditions only).
- ``'rk4'``: explicit, fourth order (Dirichlet conditions only).
- ``'crank-nicolson'``: implicit, second order, unconditionally stable and
  energy-conserving (equivalent to the Newmark method with
  :math:`\beta = 1/4, \gamma = 1/2`). Supports Neumann and Robin conditions.
- ``'backward-euler'``: implicit, first order, damps the solution.

``'forward-euler'`` is not available, since it is unstable for undamped waves.
Time-dependent boundary values and source terms work as for first-order
problems.

Stability of high-order operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On non-periodic equidistant grids, the one-sided boundary stencils of high
accuracy orders give the discrete second derivative eigenvalues that make the
solution grow exponentially, regardless of the time-stepping method: complex
eigenvalues (fatal for wave equations) from ``acc=4`` on, and eigenvalues with
positive real part (fatal for any time-dependent problem) from ``acc=6`` on.

Use ``boundary="centered"`` in that case. Points near the boundary then use
the widest centered stencil that fits into the grid, which keeps the spectrum
real and negative for any ``acc``:

.. code-block:: python

    L = c**2 * Diff(0, dx, acc=6, boundary="centered")**2

The accuracy drops towards the boundary (to second order next to it), but the
global error of time-dependent problems typically still converges with
fourth to fifth order, much better than with ``acc=2``.

On non-uniform grids, second derivatives with ``acc=2`` and ``acc=4`` are
stable for both diffusion and waves.

``TimeDependentPDE`` checks the spectrum of each second derivative in the
operator and emits a ``RuntimeWarning`` if the problem would be unstable.


Using Iterative Solvers
-------------------------

For large problems, implicit methods can use iterative solvers with
preconditioning — the same interface as :meth:`PDE.solve`:

.. code:: python

    pde = TimeDependentPDE(L, u0, bc, t)
    u = pde.solve(
        method='crank-nicolson',
        solver='gmres',
        preconditioner='ilu',
        rtol=1e-10,
    )

Supported solvers: ``'cg'``, ``'gmres'``, ``'bicgstab'``, ``'lgmres'``,
``'minres'``, or a custom callable.  See :doc:`solving-pdes` for details.


Monitoring with Callbacks
---------------------------

Pass a ``callback`` function that is called after each time step.
Return ``False`` from the callback to stop early:

.. code:: python

    def monitor(step, t, u):
        if step % 100 == 0:
            print(f"Step {step}, t={t:.4f}, max|u|={np.max(np.abs(u)):.6f}")
        if np.max(np.abs(u)) < 1e-10:
            return False  # solution has decayed, stop early

    u = pde.solve(method='crank-nicolson', callback=monitor)


Storing Solution History
-------------------------

By default, :meth:`~findiff.ivp.TimeDependentPDE.solve` returns only
the final solution to save memory.  Use ``store_every`` to keep
snapshots:

.. code:: python

    # Store every 10th time step
    sol = pde.solve(method='crank-nicolson', store_every=10)
    print(sol.t.shape)      # (num_stored_steps,)
    print(sol.u.shape)      # (num_stored_steps, *spatial_shape)
    print(sol.final.shape)  # spatial_shape

    # Store all steps
    sol = pde.solve(method='crank-nicolson', store_every=1)

    # Access individual snapshots
    u_at_step_5 = sol[5]
    t_at_step_5 = sol.t[5]

When ``store_every`` is set, the return type is a
:class:`~findiff.ivp.MOLSolution` instead of a plain *ndarray*.


Choosing Time Steps
--------------------

**Explicit methods** are subject to the CFL stability condition.  For
diffusion problems with coefficient :math:`D`:

.. math::

   \Delta t < \frac{\Delta x^2}{2\, D}

If the time step is too large, the solution will blow up.  Forward Euler
is the most restrictive; RK4 has a somewhat larger stability region.

**Implicit methods** (Backward Euler, Crank-Nicolson) are
unconditionally stable — any time step will produce a bounded solution.
However, accuracy still improves with smaller steps:

- Backward Euler: :math:`O(\Delta t)` — first-order accurate
- Crank-Nicolson: :math:`O(\Delta t^2)` — second-order accurate

For most problems, **Crank-Nicolson is the recommended choice**: it
is unconditionally stable and second-order accurate.
