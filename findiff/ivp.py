"""
Method of Lines solver for time-dependent PDEs.

Solves initial-boundary value problems of the form u_t = L(u) + s(t), or
u_tt = L(u) + s(t) for problems of second order in time (e.g. the wave
equation), where L is a spatial differential operator (Expression) and s an
optional source term.
"""

import warnings

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
    v : ndarray or None
        Snapshots of the time derivative :math:`u_t` for problems of second
        order in time, otherwise None.
    """

    def __init__(self, t, u, v=None):
        self.t = np.asarray(t)
        self.u = np.asarray(u)
        self.v = None if v is None else np.asarray(v)

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


def _inject_bc_rows(A, bc_lhs, rows):
    """Replace the rows *rows* of *A* by the same rows of *bc_lhs*.

    Same row-replacement pattern as :meth:`PDE.solve` in ``pde.py``, but done
    with sparse masks instead of a (slow) conversion to LIL format.
    """
    keep = np.ones(A.shape[0])
    keep[rows] = 0
    return sparse.csr_matrix(
        sparse.diags(keep) @ A + sparse.diags(1 - keep) @ sparse.csr_matrix(bc_lhs)
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
        self._A = _inject_bc_rows(A, self.bcs.lhs, self._bc_rows)
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
# Time steppers for problems of second order in time: u_tt = L(u) + s(t)
# The state is the pair (u, v) with v = u_t.
# ---------------------------------------------------------------------------

class _VerletStepper(_ExplicitStepper):
    r"""Stoermer-Verlet (velocity form), explicit, second order, symplectic.

    .. math::
        v^{n+1/2} = v^n + \tfrac{\Delta t}{2} a^n, \quad
        u^{n+1} = u^n + \Delta t\, v^{n+1/2}, \quad
        v^{n+1} = v^{n+1/2} + \tfrac{\Delta t}{2} a^{n+1}

    with :math:`a = L(u) + s(t)`.
    """

    def __init__(self, operator, shape, bcs, source=None):
        super().__init__(operator, shape, bcs, source)
        self._last = None  # (t, u, a) to reuse the acceleration of the last step

    def step(self, state, t, dt):
        u, v = state
        if self._last is not None and self._last[0] == t and self._last[1] is u:
            a = self._last[2]
        else:
            a = self._rate(u, t)
        v_half = v + 0.5 * dt * a
        u_new = self._enforce_bc(u + dt * v_half, t + dt)
        a_new = self._rate(u_new, t + dt)
        v_new = v_half + 0.5 * dt * a_new
        self._last = (t + dt, u_new, a_new)
        return u_new, v_new


class _SecondOrderRK4Stepper(_ExplicitStepper):
    """Classic RK4 applied to the first-order system u_t = v, v_t = L(u) + s."""

    def step(self, state, t, dt):
        u, v = state
        t_half = t + 0.5 * dt
        k1u, k1v = v, self._rate(u, t)
        u2 = self._enforce_bc(u + 0.5 * dt * k1u, t_half)
        k2u, k2v = v + 0.5 * dt * k1v, self._rate(u2, t_half)
        u3 = self._enforce_bc(u + 0.5 * dt * k2u, t_half)
        k3u, k3v = v + 0.5 * dt * k2v, self._rate(u3, t_half)
        u4 = self._enforce_bc(u + dt * k3u, t + dt)
        k4u, k4v = v + dt * k3v, self._rate(u4, t + dt)
        u_new = u + (dt / 6.0) * (k1u + 2 * k2u + 2 * k3u + k4u)
        v_new = v + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        return self._enforce_bc(u_new, t + dt), v_new


class _SecondOrderImplicitStepper(_ImplicitStepper):
    r"""Theta method applied to the first-order system u_t = v, v_t = L(u) + s.

    For :math:`\theta = 1/2` (Crank-Nicolson) this is equivalent to the
    Newmark method with :math:`\beta = 1/4, \gamma = 1/2` (average
    acceleration): unconditionally stable and energy-conserving.
    Boundary conditions replace the rows of the *u* equations.
    """

    def step(self, state, t, dt):
        if self._cached_dt is None or not np.isclose(
            dt, self._cached_dt, rtol=self._DT_RTOL, atol=0
        ):
            self._build_system(dt)
            self._cached_dt = dt

        u, v = state
        N = u.size
        rhs = self._B.dot(np.concatenate([u.reshape(-1), v.reshape(-1)]))
        if self.source is not None:
            s_new = self.source(t + dt).reshape(-1)
            if self.theta < 1.0:
                s_old = self.source(t).reshape(-1)
                rhs[N:] += dt * (self.theta * s_new + (1.0 - self.theta) * s_old)
            else:
                rhs[N:] += dt * s_new
        rhs[self._bc_rows] = self._bc_values(t + dt)

        y = self._solve_linear_system(self._A, rhs)
        return y[:N].reshape(self.shape), y[N:].reshape(self.shape)

    def _build_system(self, dt):
        L = self._L_mat
        N = L.shape[0]
        I = sparse.eye(N, format='csr')
        th = self.theta

        A = sparse.bmat([[I, -th * dt * I], [-th * dt * L, I]], format='csr')
        bc_lhs = sparse.block_diag(
            [sparse.csr_matrix(self.bcs.lhs), sparse.csr_matrix((N, N))],
            format='csr',
        )
        self._A = _inject_bc_rows(A, bc_lhs, self._bc_rows)
        self._lu = None
        self._B = sparse.bmat(
            [[I, (1 - th) * dt * I], [(1 - th) * dt * L, I]], format='csr'
        )


class _SecondOrderBackwardEulerStepper(_SecondOrderImplicitStepper):
    def __init__(self, operator, shape, bcs, solver, solver_options, source=None):
        super().__init__(operator, shape, bcs, theta=1.0,
                         solver=solver, solver_options=solver_options,
                         source=source)


class _SecondOrderCrankNicolsonStepper(_SecondOrderImplicitStepper):
    def __init__(self, operator, shape, bcs, solver, solver_options, source=None):
        super().__init__(operator, shape, bcs, theta=0.5,
                         solver=solver, solver_options=solver_options,
                         source=source)


# ---------------------------------------------------------------------------
# Stability check of the spatial operator
# ---------------------------------------------------------------------------

def _iter_diffs(expr):
    from findiff.operators import Diff

    if isinstance(expr, Diff):
        yield expr
    for child in getattr(expr, "children", []):
        yield from _iter_diffs(child)
    for part in getattr(expr, "_parts", []):  # e.g. Laplacian
        yield from _iter_diffs(part)


def _warn_if_unstable(operator, second_order, shape):
    """Warns if a second derivative in *operator* makes the problem unstable.

    One-sided high-order boundary stencils can give the discrete second
    derivative eigenvalues with positive real part (unstable for any
    time-dependent problem) or complex eigenvalues (unstable for problems of
    second order in time). This is checked on the 1D operator along each
    axis with Dirichlet conditions, on at most `_CHECK_POINTS` grid points.
    """
    from findiff.grids import EquidistantAxis, NonEquidistantAxis
    from findiff.operators import Diff

    for d in _iter_diffs(operator):
        axis = d.axis
        if d.order != 2 or axis is None or axis.periodic or d.dim >= len(shape):
            continue
        n = min(shape[d.dim], _CHECK_POINTS)
        if isinstance(axis, EquidistantAxis):
            axis_1d = EquidistantAxis(0, axis.spacing)
        elif isinstance(axis, NonEquidistantAxis) and len(axis.coords) <= _CHECK_POINTS:
            axis_1d, n = NonEquidistantAxis(0, axis.coords), len(axis.coords)
        else:
            continue
        if n < 5:
            continue

        d_1d = Diff(0, axis_1d, acc=d.acc, scheme=d.scheme,
                    boundary=getattr(d, "boundary", "one-sided"))
        d_1d._order = 2
        try:
            ev = np.linalg.eigvals(d_1d.matrix((n,)).toarray()[1:-1, 1:-1])
        except Exception:  # the check must never break a solve
            continue
        scale = np.abs(ev).max()
        growing = ev.real.max() > 1e-6 * scale
        oscillating = second_order and np.abs(ev.imag).max() > 1e-6 * scale
        if growing or oscillating:
            reason = ("eigenvalues with positive real part" if growing
                      else "complex eigenvalues")
            warnings.warn(
                f"The discrete second derivative along axis {d.dim} (acc={d.acc}"
                f"{', compact scheme' if d.scheme is not None else ''}) has {reason} "
                f"due to its boundary stencils, which makes this problem unstable. "
                f"Use Diff(..., boundary='centered') or a lower acc.",
                RuntimeWarning,
                stacklevel=3,
            )
            return


_CHECK_POINTS = 201


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TimeDependentPDE:
    r"""Solve a time-dependent PDE using the Method of Lines.

    Solves problems of the form

    .. math::

        \frac{\partial u}{\partial t} = L(u) + s(t)

    or, if the initial velocity ``v0`` is given, problems of second order in
    time such as the wave equation,

    .. math::

        \frac{\partial^2 u}{\partial t^2} = L(u) + s(t)

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
    v0 : ndarray or None
        Initial velocity :math:`u_t(t_0)`. If given, the problem is of
        second order in time.

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

    _second_order_methods = {
        'verlet': _VerletStepper,
        'rk4': _SecondOrderRK4Stepper,
        'backward-euler': _SecondOrderBackwardEulerStepper,
        'crank-nicolson': _SecondOrderCrankNicolsonStepper,
    }

    def __init__(self, lhs, u0, bcs, t, source=None, v0=None):
        self.lhs = lhs
        self.u0 = np.asarray(u0, dtype=float)
        self.bcs = bcs
        self.t = np.asarray(t, dtype=float)
        self.source = _make_source(source, self.u0.shape)
        self.v0 = None if v0 is None else np.asarray(v0, dtype=float)
        if self.v0 is not None and self.v0.shape != self.u0.shape:
            raise ValueError(
                f"v0 shape {self.v0.shape} does not match u0 shape {self.u0.shape}"
            )
        _warn_if_unstable(lhs, self.v0 is not None, self.u0.shape)

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
            - ``'verlet'``: explicit second-order Stoermer-Verlet, only for
              problems of second order in time (``v0`` given).

            For problems of second order in time, ``'forward-euler'`` is not
            available (it is unstable for undamped waves); the other methods
            are applied to the equivalent first-order system.

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
            Solution container (when ``store_every`` is set). For problems
            of second order in time it also holds the velocities ``v``.
        """
        second_order = self.v0 is not None
        methods = self._second_order_methods if second_order else self._methods
        if method not in methods:
            if second_order and method == 'forward-euler':
                raise ValueError(
                    "'forward-euler' is unstable for problems of second order "
                    "in time; use 'verlet', 'rk4' or 'crank-nicolson'."
                )
            if not second_order and method == 'verlet':
                raise ValueError(
                    "'verlet' requires a problem of second order in time "
                    "(pass the initial velocity v0)."
                )
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Supported: {sorted(methods.keys())}"
            )

        shape = self.bcs.shape
        stepper_cls = methods[method]

        if method in ('forward-euler', 'rk4', 'verlet'):
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

        v = self.v0.copy() if second_order else None

        if store_every is not None:
            stored_t = [self.t[0]]
            stored_u = [u.copy()]
            stored_v = [v.copy()] if second_order else None

        def store(t_i):
            stored_t.append(t_i)
            stored_u.append(u.copy())
            if second_order:
                stored_v.append(v.copy())

        def solution():
            return MOLSolution(stored_t, stored_u, stored_v)

        for i in range(1, nt):
            dt = self.t[i] - self.t[i - 1]
            if second_order:
                u, v = stepper.step((u, v), self.t[i - 1], dt)
            else:
                u = stepper.step(u, self.t[i - 1], dt)

            if store_every is not None and i % store_every == 0:
                store(self.t[i])

            if callback is not None:
                result = callback(i, self.t[i], u)
                if result is False:
                    if store_every is not None:
                        # Ensure the last computed step is stored
                        if stored_t[-1] != self.t[i]:
                            store(self.t[i])
                        return solution()
                    return u.copy()

        if store_every is not None:
            # Ensure the final step is always stored
            if stored_t[-1] != self.t[-1]:
                store(self.t[-1])
            return solution()

        return u.copy()
