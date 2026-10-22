"""
Method of Lines solver for time-dependent PDEs.

Solves initial-boundary value problems of the form u_t = L(u) + s(t)
where L is a spatial differential operator (Expression) and s an optional
source term.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


class MOLSolution:
    """Container for time-dependent PDE solutions.

    Returned by :meth:`TimeDependentPDE.solve` when ``store_every`` is set.

    Attributes
    ----------
    t : ndarray, shape (nt_stored,)
        Time points at which the solution is stored.
    u : ndarray, shape (nt_stored, *spatial_shape)
        Solution snapshots.  ``u[i]`` is the solution at time ``t[i]``.
    """

    def __init__(self, t, u):
        self.t = np.asarray(t)
        self.u = np.asarray(u)

    def __getitem__(self, idx):
        """Index into time steps: ``solution[i]`` returns *u* at time ``t[i]``."""
        return self.u[idx]

    @property
    def final(self):
        """Solution at the final time."""
        return self.u[-1]


# ---------------------------------------------------------------------------
# Boundary-condition helpers
# ---------------------------------------------------------------------------

def _get_dirichlet_data(bcs):
    """Extract flat indices and values of Dirichlet boundary conditions.

    A row in ``bcs.lhs`` is considered Dirichlet when it contains a single
    non-zero entry equal to 1.0 on the diagonal.

    Returns
    -------
    inds : ndarray of int
    vals : ndarray of float
    """
    nz_rows = np.unique(list(bcs.row_inds()))
    if len(nz_rows) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    lhs_csr = sparse.csr_matrix(bcs.lhs)
    rhs_arr = np.asarray(bcs.rhs.toarray()).ravel()

    dirichlet_inds = []
    dirichlet_vals = []
    for i in nz_rows:
        row_start = lhs_csr.indptr[i]
        row_end = lhs_csr.indptr[i + 1]
        cols = lhs_csr.indices[row_start:row_end]
        data = lhs_csr.data[row_start:row_end]
        if len(cols) == 1 and cols[0] == i and abs(data[0] - 1.0) < 1e-14:
            dirichlet_inds.append(i)
            dirichlet_vals.append(rhs_arr[i])
    return np.array(dirichlet_inds, dtype=int), np.array(dirichlet_vals)


class _BCValues:
    """Boundary values for given flat row indices, evaluated at time t."""

    def __init__(self, bcs, rows):
        self.bcs = bcs
        self.rows = np.asarray(rows, dtype=int)
        self._static = None
        if not getattr(bcs, "is_time_dependent", False):
            self._static = bcs.values_at(self.rows, None)

    def __call__(self, t):
        if self._static is not None:
            return self._static
        return self.bcs.values_at(self.rows, t)


def _make_source(source, shape):
    """Normalizes the source term to None or a callable s(t) -> array of `shape`."""
    if source is None:
        return None
    if callable(source):
        return lambda t: np.broadcast_to(np.asarray(source(t), dtype=float), shape)
    values = np.broadcast_to(np.asarray(source, dtype=float), shape)
    return lambda t: values


def _apply_dirichlet(u, dirichlet_inds, dirichlet_vals):
    """Overwrite Dirichlet boundary values in *u* (in-place)."""
    if len(dirichlet_inds) == 0:
        return u
    shape = u.shape
    u_flat = u.reshape(-1)
    u_flat[dirichlet_inds] = dirichlet_vals
    return u_flat.reshape(shape)


def _bc_rows(bcs):
    """Row indices of all boundary-condition equations."""
    return np.unique(np.fromiter(bcs.row_inds(), dtype=int))


def _inject_bc_rows(A, bcs, rows):
    """Replace the rows *rows* of *A* by the boundary-condition equations.

    Same row-replacement pattern as :meth:`PDE.solve` in ``pde.py``, but done
    with sparse masks instead of a (slow) conversion to LIL format.
    """
    keep = np.ones(A.shape[0])
    keep[rows] = 0
    return sparse.csr_matrix(
        sparse.diags(keep) @ A + sparse.diags(1 - keep) @ sparse.csr_matrix(bcs.lhs)
    )


# ---------------------------------------------------------------------------
# Explicit time steppers
# ---------------------------------------------------------------------------

class _ExplicitStepper:
    """Base class for explicit time-stepping methods."""

    def __init__(self, operator, shape, bcs, source=None):
        self.operator = operator
        self.shape = shape
        self.source = source
        self.dirichlet_inds, _ = _get_dirichlet_data(bcs)
        self._bc_values = _BCValues(bcs, self.dirichlet_inds)

        # Verify all BCs are Dirichlet
        all_bc_rows = np.unique(list(bcs.row_inds()))
        if len(all_bc_rows) != len(self.dirichlet_inds):
            raise ValueError(
                "Explicit methods (forward-euler, rk4) require all boundary "
                "conditions to be Dirichlet. For Neumann or Robin BCs, use "
                "an implicit method (backward-euler, crank-nicolson)."
            )

    def _enforce_bc(self, u, t):
        return _apply_dirichlet(u, self.dirichlet_inds, self._bc_values(t))

    def _rate(self, u, t):
        rate = self.operator(u)
        if self.source is not None:
            rate = rate + self.source(t)
        return rate

    def step(self, u, t, dt):
        raise NotImplementedError


class _ForwardEulerStepper(_ExplicitStepper):
    r"""Forward Euler: :math:`u^{n+1} = u^n + \Delta t\, (L(u^n) + s(t^n))`."""

    def step(self, u, t, dt):
        u_new = u + dt * self._rate(u, t)
        return self._enforce_bc(u_new, t + dt)


class _RK4Stepper(_ExplicitStepper):
    """Classic fourth-order Runge-Kutta."""

    def step(self, u, t, dt):
        t_half = t + 0.5 * dt
        k1 = self._rate(u, t)
        k2 = self._rate(self._enforce_bc(u + 0.5 * dt * k1, t_half), t_half)
        k3 = self._rate(self._enforce_bc(u + 0.5 * dt * k2, t_half), t_half)
        k4 = self._rate(self._enforce_bc(u + dt * k3, t + dt), t + dt)
        u_new = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return self._enforce_bc(u_new, t + dt)


# ---------------------------------------------------------------------------
# Implicit time steppers
# ---------------------------------------------------------------------------

class _ImplicitStepper:
    r"""Base class for implicit time-stepping methods.

    Solves :math:`(I - \theta\,\Delta t\, L)\, u^{n+1}
    = (I + (1-\theta)\,\Delta t\, L)\, u^n` with boundary-condition
    row replacement.
    """

    # Time steps that differ only by round-off (e.g. from np.linspace) reuse
    # the assembled and factorized system.
    _DT_RTOL = 1e-10

    def __init__(self, operator, shape, bcs, theta, solver, solver_options,
                 source=None):
        self.shape = shape
        self.bcs = bcs
        self.theta = theta
        self.source = source
        self.solver = solver
        self.solver_options = dict(solver_options)
        self._L_mat = sparse.csr_matrix(operator.matrix(shape))
        self._bc_rows = _bc_rows(bcs)
        self._bc_values = _BCValues(bcs, self._bc_rows)
        self._cached_dt = None
        self._A = None   # (I - theta*dt*L) with BCs
        self._B = None   # (I + (1-theta)*dt*L)
        self._lu = None  # LU factorization of A for the direct solver

    def step(self, u, t, dt):
        if self._cached_dt is None or not np.isclose(
            dt, self._cached_dt, rtol=self._DT_RTOL, atol=0
        ):
            self._build_system(dt)
            self._cached_dt = dt

        rhs = self._B.dot(u.reshape(-1))
        if self.source is not None:
            s_new = self.source(t + dt).reshape(-1)
            if self.theta < 1.0:
                s_old = self.source(t).reshape(-1)
                rhs += dt * (self.theta * s_new + (1.0 - self.theta) * s_old)
            else:
                rhs += dt * s_new
        rhs[self._bc_rows] = self._bc_values(t + dt)

        x = self._solve_linear_system(self._A, rhs)
        return x.reshape(self.shape)

    def _build_system(self, dt):
        N = self._L_mat.shape[0]
        I = sparse.eye(N, format='csr')

        # LHS: I - theta*dt*L, with BC rows replaced
        A = I - self.theta * dt * self._L_mat
        self._A = _inject_bc_rows(A, self.bcs, self._bc_rows)
        self._lu = None

        # RHS matrix: I + (1-theta)*dt*L
        if self.theta < 1.0:
            self._B = I + (1.0 - self.theta) * dt * self._L_mat
        else:
            self._B = I

    def _solve_linear_system(self, A, b):
        if self.solver is None or self.solver == 'direct':
            if self._lu is None:
                self._lu = splu(sparse.csc_matrix(A))
            return self._lu.solve(b)

        opts = dict(self.solver_options)

        # Preconditioner shorthand
        preconditioner = opts.pop('preconditioner', None)
        if preconditioner == 'ilu':
            from scipy.sparse.linalg import spilu, LinearOperator
            ilu = spilu(sparse.csc_matrix(A))
            opts['M'] = LinearOperator(A.shape, matvec=ilu.solve)
        elif preconditioner is not None:
            opts['M'] = preconditioner

        if callable(self.solver):
            result = self.solver(A, b, **opts)
            if isinstance(result, tuple):
                x, info = result
                if info != 0:
                    raise RuntimeError(
                        f"Custom solver did not converge (info={info})"
                    )
                return x
            return result

        solve_func = self._get_iterative_solver(self.solver)
        x, info = solve_func(A, b, **opts)
        if info > 0:
            raise RuntimeError(
                f"Solver '{self.solver}' did not converge within "
                f"the maximum number of iterations"
            )
        elif info < 0:
            raise RuntimeError(
                f"Solver '{self.solver}' encountered an illegal input "
                f"or breakdown (info={info})"
            )
        return x

    @staticmethod
    def _get_iterative_solver(name):
        from scipy.sparse.linalg import cg, gmres, bicgstab, lgmres, minres
        solvers = {
            'cg': cg,
            'gmres': gmres,
            'bicgstab': bicgstab,
            'lgmres': lgmres,
            'minres': minres,
        }
        if name not in solvers:
            raise ValueError(
                f"Unknown solver '{name}'. "
                f"Supported: {sorted(solvers.keys())}"
            )
        return solvers[name]


class _BackwardEulerStepper(_ImplicitStepper):
    r"""Backward Euler: :math:`(I - \Delta t\, L)\, u^{n+1} = u^n + \Delta t\, s^{n+1}`."""

    def __init__(self, operator, shape, bcs, solver, solver_options, source=None):
        super().__init__(operator, shape, bcs, theta=1.0,
                         solver=solver, solver_options=solver_options,
                         source=source)


class _CrankNicolsonStepper(_ImplicitStepper):
    r"""Crank-Nicolson: :math:`(I - \tfrac{1}{2}\Delta t\, L)\, u^{n+1}
    = (I + \tfrac{1}{2}\Delta t\, L)\, u^n`."""

    def __init__(self, operator, shape, bcs, solver, solver_options, source=None):
        super().__init__(operator, shape, bcs, theta=0.5,
                         solver=solver, solver_options=solver_options,
                         source=source)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TimeDependentPDE:
    r"""Solve a time-dependent PDE using the Method of Lines.

    Solves problems of the form

    .. math::

        \frac{\partial u}{\partial t} = L(u) + s(t)

    where *L* is a spatial differential operator built with *findiff* and
    *s* an optional source term.

    Parameters
    ----------
    lhs : Expression
        The spatial differential operator *L*.
    u0 : ndarray
        Initial condition array with the spatial grid shape.
    bcs : BoundaryConditions
        Spatial boundary conditions applied at each time step. Boundary
        values may be callables ``g(t)`` for time-dependent conditions.
    t : 1D ndarray
        Time points.  ``t[0]`` is the initial time.
    source : ndarray, callable or None
        Optional source term *s*: an array (constant in time, broadcastable
        to the grid shape) or a callable ``s(t)`` returning such an array.

    Examples
    --------
    1D heat equation :math:`u_t = D\, u_{xx}`:

        >>> import numpy as np
        >>> from findiff import Diff, BoundaryConditions
        >>> from findiff.ivp import TimeDependentPDE
        >>> nx = 51
        >>> x = np.linspace(0, 1, nx)
        >>> dx = x[1] - x[0]
        >>> L = 0.01 * Diff(0, dx)**2
        >>> u0 = np.sin(np.pi * x)
        >>> bc = BoundaryConditions((nx,))
        >>> bc[0] = 0
        >>> bc[-1] = 0
        >>> t = np.linspace(0, 0.1, 100)
        >>> u_final = TimeDependentPDE(L, u0, bc, t).solve()
        >>> u_final.shape
        (51,)
    """

    _methods = {
        'forward-euler': _ForwardEulerStepper,
        'rk4': _RK4Stepper,
        'backward-euler': _BackwardEulerStepper,
        'crank-nicolson': _CrankNicolsonStepper,
    }

    def __init__(self, lhs, u0, bcs, t, source=None):
        self.lhs = lhs
        self.u0 = np.asarray(u0, dtype=float)
        self.bcs = bcs
        self.t = np.asarray(t, dtype=float)
        self.source = _make_source(source, self.u0.shape)

        if self.t.ndim != 1 or len(self.t) < 2:
            raise ValueError("t must be a 1D array with at least 2 points")
        if self.u0.shape != bcs.shape:
            raise ValueError(
                f"u0 shape {self.u0.shape} does not match "
                f"BoundaryConditions shape {bcs.shape}"
            )

    def solve(self, method='crank-nicolson', solver=None,
              store_every=None, callback=None, **solver_options):
        """Advance the solution in time.

        Parameters
        ----------
        method : str
            Time integration method:

            - ``'forward-euler'``: explicit first-order.
            - ``'rk4'``: explicit fourth-order Runge-Kutta.
            - ``'backward-euler'``: implicit first-order.
            - ``'crank-nicolson'``: implicit second-order (default).

        solver : str, callable, or None
            Linear solver for implicit methods.  Same interface as
            :meth:`PDE.solve`.  Ignored for explicit methods.
        store_every : int or None
            If ``None`` (default), return only the final solution as an
            *ndarray*.  If an integer *N*, store every *N*-th time step
            and return a :class:`MOLSolution`.  ``store_every=1`` stores
            all steps.
        callback : callable or None
            Called as ``callback(step, t_i, u_i)`` after each time step.
            Return ``False`` to stop early.
        **solver_options
            Passed to the linear solver (``rtol``, ``maxiter``,
            ``preconditioner``, etc.).

        Returns
        -------
        ndarray
            Final solution (when ``store_every`` is ``None``).
        MOLSolution
            Solution container (when ``store_every`` is set).
        """
        if method not in self._methods:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Supported: {sorted(self._methods.keys())}"
            )

        shape = self.bcs.shape
        stepper_cls = self._methods[method]

        if method in ('forward-euler', 'rk4'):
            if solver is not None or solver_options:
                raise ValueError(
                    f"solver and solver_options are not used with "
                    f"explicit method '{method}'"
                )
            stepper = stepper_cls(self.lhs, shape, self.bcs, source=self.source)
        else:
            stepper = stepper_cls(
                self.lhs, shape, self.bcs, solver, solver_options,
                source=self.source,
            )

        # Time-stepping loop
        nt = len(self.t)
        u = self.u0.copy()
        dirichlet_inds, _ = _get_dirichlet_data(self.bcs)
        u = _apply_dirichlet(
            u, dirichlet_inds, _BCValues(self.bcs, dirichlet_inds)(self.t[0])
        )

        if store_every is not None:
            stored_t = [self.t[0]]
            stored_u = [u.copy()]

        for i in range(1, nt):
            dt = self.t[i] - self.t[i - 1]
            u = stepper.step(u, self.t[i - 1], dt)

            if store_every is not None and i % store_every == 0:
                stored_t.append(self.t[i])
                stored_u.append(u.copy())

            if callback is not None:
                result = callback(i, self.t[i], u)
                if result is False:
                    if store_every is not None:
                        # Ensure the last computed step is stored
                        if stored_t[-1] != self.t[i]:
                            stored_t.append(self.t[i])
                            stored_u.append(u.copy())
                        return MOLSolution(stored_t, stored_u)
                    return u.copy()

        if store_every is not None:
            # Ensure the final step is always stored
            if stored_t[-1] != self.t[-1]:
                stored_t.append(self.t[-1])
                stored_u.append(u.copy())
            return MOLSolution(stored_t, stored_u)

        return u.copy()
