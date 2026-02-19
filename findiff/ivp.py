"""
Method of Lines solver for time-dependent PDEs.

Solves initial-boundary value problems of the form u_t = L(u)
where L is a spatial differential operator (Expression).
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


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


def _apply_dirichlet(u, dirichlet_inds, dirichlet_vals):
    """Overwrite Dirichlet boundary values in *u* (in-place)."""
    if len(dirichlet_inds) == 0:
        return u
    shape = u.shape
    u_flat = u.reshape(-1)
    u_flat[dirichlet_inds] = dirichlet_vals
    return u_flat.reshape(shape)


def _inject_bcs(A, rhs_vec, bcs):
    """Replace rows in *A* and *rhs_vec* with boundary-condition equations.

    Follows the same row-replacement pattern used by
    :meth:`PDE.solve` in ``pde.py``.
    """
    A_lil = sparse.lil_matrix(A)
    rhs = rhs_vec.copy()
    nz = list(bcs.row_inds())
    A_lil[nz, :] = bcs.lhs[nz, :]
    rhs[nz] = np.asarray(bcs.rhs[nz].toarray()).ravel()
    return sparse.csr_matrix(A_lil), rhs


# ---------------------------------------------------------------------------
# Explicit time steppers
# ---------------------------------------------------------------------------

class _ExplicitStepper:
    """Base class for explicit time-stepping methods."""

    def __init__(self, operator, shape, bcs):
        self.operator = operator
        self.shape = shape
        self.dirichlet_inds, self.dirichlet_vals = _get_dirichlet_data(bcs)

        # Verify all BCs are Dirichlet
        all_bc_rows = np.unique(list(bcs.row_inds()))
        if len(all_bc_rows) != len(self.dirichlet_inds):
            raise ValueError(
                "Explicit methods (forward-euler, rk4) require all boundary "
                "conditions to be Dirichlet. For Neumann or Robin BCs, use "
                "an implicit method (backward-euler, crank-nicolson)."
            )

    def _enforce_bc(self, u):
        return _apply_dirichlet(u, self.dirichlet_inds, self.dirichlet_vals)

    def step(self, u, dt):
        raise NotImplementedError


class _ForwardEulerStepper(_ExplicitStepper):
    r"""Forward Euler: :math:`u^{n+1} = u^n + \Delta t\, L(u^n)`."""

    def step(self, u, dt):
        u_new = u + dt * self.operator(u)
        return self._enforce_bc(u_new)


class _RK4Stepper(_ExplicitStepper):
    """Classic fourth-order Runge-Kutta."""

    def step(self, u, dt):
        L = self.operator
        k1 = L(u)
        k2 = L(self._enforce_bc(u + 0.5 * dt * k1))
        k3 = L(self._enforce_bc(u + 0.5 * dt * k2))
        k4 = L(self._enforce_bc(u + dt * k3))
        u_new = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return self._enforce_bc(u_new)


# ---------------------------------------------------------------------------
# Implicit time steppers
# ---------------------------------------------------------------------------

class _ImplicitStepper:
    r"""Base class for implicit time-stepping methods.

    Solves :math:`(I - \theta\,\Delta t\, L)\, u^{n+1}
    = (I + (1-\theta)\,\Delta t\, L)\, u^n` with boundary-condition
    row replacement.
    """

    def __init__(self, operator, shape, bcs, theta, solver, solver_options):
        self.shape = shape
        self.bcs = bcs
        self.theta = theta
        self.solver = solver
        self.solver_options = dict(solver_options)
        self._L_mat = operator.matrix(shape)
        self._cached_dt = None
        self._A = None   # (I - theta*dt*L) with BCs
        self._B = None   # (I + (1-theta)*dt*L)

    def step(self, u, dt):
        if self._cached_dt != dt:
            self._build_system(dt)
            self._cached_dt = dt

        rhs = self._B.dot(u.reshape(-1))
        _, rhs = _inject_bcs(self._A, rhs, self.bcs)

        x = self._solve_linear_system(self._A, rhs)
        return x.reshape(self.shape)

    def _build_system(self, dt):
        N = self._L_mat.shape[0]
        I = sparse.eye(N, format='csr')

        # LHS: I - theta*dt*L
        A = I - self.theta * dt * self._L_mat

        # Inject BCs into A (row replacement)
        A_lil = sparse.lil_matrix(A)
        nz = list(self.bcs.row_inds())
        A_lil[nz, :] = self.bcs.lhs[nz, :]
        self._A = sparse.csr_matrix(A_lil)

        # RHS matrix: I + (1-theta)*dt*L
        if self.theta < 1.0:
            self._B = I + (1.0 - self.theta) * dt * self._L_mat
        else:
            self._B = I

    def _solve_linear_system(self, A, b):
        if self.solver is None or self.solver == 'direct':
            return spsolve(A, b)

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
    r"""Backward Euler: :math:`(I - \Delta t\, L)\, u^{n+1} = u^n`."""

    def __init__(self, operator, shape, bcs, solver, solver_options):
        super().__init__(operator, shape, bcs, theta=1.0,
                         solver=solver, solver_options=solver_options)


class _CrankNicolsonStepper(_ImplicitStepper):
    r"""Crank-Nicolson: :math:`(I - \tfrac{1}{2}\Delta t\, L)\, u^{n+1}
    = (I + \tfrac{1}{2}\Delta t\, L)\, u^n`."""

    def __init__(self, operator, shape, bcs, solver, solver_options):
        super().__init__(operator, shape, bcs, theta=0.5,
                         solver=solver, solver_options=solver_options)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TimeDependentPDE:
    r"""Solve a time-dependent PDE using the Method of Lines.

    Solves problems of the form

    .. math::

        \frac{\partial u}{\partial t} = L(u)

    where *L* is a spatial differential operator built with *findiff*.

    Parameters
    ----------
    lhs : Expression
        The spatial differential operator *L*.
    u0 : ndarray
        Initial condition array with the spatial grid shape.
    bcs : BoundaryConditions
        Spatial boundary conditions applied at each time step.
    t : 1D ndarray
        Time points.  ``t[0]`` is the initial time.

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

    def __init__(self, lhs, u0, bcs, t):
        self.lhs = lhs
        self.u0 = np.asarray(u0, dtype=float)
        self.bcs = bcs
        self.t = np.asarray(t, dtype=float)

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
            stepper = stepper_cls(self.lhs, shape, self.bcs)
        else:
            stepper = stepper_cls(
                self.lhs, shape, self.bcs, solver, solver_options
            )

        # Time-stepping loop
        nt = len(self.t)
        u = self.u0.copy()

        if store_every is not None:
            stored_t = [self.t[0]]
            stored_u = [u.copy()]

        for i in range(1, nt):
            dt = self.t[i] - self.t[i - 1]
            u = stepper.step(u, dt)

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
