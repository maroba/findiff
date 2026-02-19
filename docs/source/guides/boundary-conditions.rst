===================
Boundary Conditions
===================

The ``BoundaryConditions`` class collects all boundary constraints for a
PDE problem. *findiff* supports Dirichlet, Neumann and Robin (mixed)
boundary conditions in any number of dimensions.

Setup
-----

.. code:: python

    import numpy as np
    from findiff import Diff, PDE, BoundaryConditions

    # 1D grid
    x = np.linspace(0, 1, 100)
    dx = x[1] - x[0]

    # 2D grid
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing='ij')


Creating a BoundaryConditions Object
--------------------------------------

Pass the grid shape to the constructor:

.. code:: python

    # 1D
    bc = BoundaryConditions((100,))

    # 2D
    bc = BoundaryConditions((50, 50))


Dirichlet Boundary Conditions
------------------------------

Dirichlet conditions fix the function value at the boundary. Assign a
scalar or array directly:

.. code:: python

    # 1D: u(0) = 0, u(1) = 1
    bc = BoundaryConditions((100,))
    bc[0] = 0
    bc[-1] = 1

    # 2D: fix values on edges
    bc = BoundaryConditions((50, 50))
    bc[0, :] = 0          # u = 0 at x = 0
    bc[-1, :] = 1         # u = 1 at x = 1
    bc[:, 0] = Y[:, 0]    # u = y at y = 0
    bc[:, -1] = Y[:, -1]  # u = y at y = 1

Boundary slicing uses standard NumPy indexing. The assigned value can be:

- A **scalar** (applied to every point on that boundary)
- A **1D array** matching the boundary size


Neumann Boundary Conditions
-----------------------------

Neumann conditions fix a derivative at the boundary. Pass a 2-tuple of
``(derivative_operator, value)``:

.. code:: python

    bc = BoundaryConditions((100,))
    bc[0] = 0                         # Dirichlet: u(0) = 0
    bc[-1] = Diff(0, dx), 0           # Neumann: u'(1) = 0

In 2D, use the normal derivative operator for each edge:

.. code:: python

    bc = BoundaryConditions((50, 50))
    bc[0, :] = Diff(0, dx), 0       # du/dx = 0 at x = 0
    bc[-1, :] = 1.0                  # Dirichlet at x = 1
    bc[:, 0] = 300.0                 # Dirichlet at y = 0
    bc[1:-1, -1] = Diff(1, dy), 0   # du/dy = 0 at y = 1

.. note::

    When specifying Neumann conditions on a partial slice like ``bc[1:-1, -1]``,
    make sure the corners are covered by other conditions so the system is
    not under-determined.


Robin (Mixed) Boundary Conditions
----------------------------------

Robin conditions combine Dirichlet and Neumann:

.. math::

   \alpha \, u + \beta \, \frac{\partial u}{\partial n} = g

Specify them as a 4-tuple ``(alpha, diff_op, beta, g)``:

.. code:: python

    bc = BoundaryConditions((100,))
    bc[0] = 1.0                              # Dirichlet: u(0) = 1
    bc[-1] = (1, Diff(0, dx), 1, 3)          # Robin: u + u' = 3 at x = 1

For 2D problems:

.. code:: python

    bc = BoundaryConditions((50, 50))
    bc[-1, :] = (1, Diff(0, dx), 0.5, g_edge)   # alpha*u + beta*du/dx = g
    bc[0, :] = u_left
    bc[:, 0] = u_bottom
    bc[:, -1] = u_top

Alternatively, construct the Robin operator yourself using ``Identity()``
and pass it as a Neumann-style 2-tuple:

.. code:: python

    from findiff import Identity

    robin_op = alpha * Identity() + beta * Diff(0, dx)
    bc[-1] = robin_op, g


Summary of Syntax
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Type
     - Syntax
     - Example
   * - Dirichlet
     - ``bc[slice] = value``
     - ``bc[0] = 0``
   * - Neumann
     - ``bc[slice] = (Diff(...), value)``
     - ``bc[-1] = Diff(0, dx), 0``
   * - Robin
     - ``bc[slice] = (alpha, Diff(...), beta, g)``
     - ``bc[-1] = (1, Diff(0, dx), 1, 3)``
