"""
This module contains class for solving Partial Differential Equations (PDE)
with Dirichlet, Neumann and Robin Boundary Conditions.
"""


import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve


class PDE:
    """
    Representation of a partial differential equation.
    """

    def __init__(self, lhs, rhs, bcs):
        """
        Initializes the PDE.

        You need to specify the left hand side (lhs) in terms of derivatives
        as well as the right hand side in terms of an array.

        Parameters
        ----------
        lhs: FinDiff object or combination of FinDiff objects
            the left hand side of the PDE
        rhs: numpy.ndarray
            the right hand side of the PDE
        bcs: BoundaryConditions
            the boundary conditions for the PDE

        """
        self.lhs = lhs
        self.rhs = rhs
        self.bcs = bcs
        self._L = None

    def solve(self, solver=None, **solver_options):
        """
        Solves the PDE.

        Parameters
        ----------
        solver : str, callable, or None
            Which linear solver to use.

            - ``None`` or ``'direct'``: use ``scipy.sparse.linalg.spsolve``
              (default, backward compatible).
            - ``'cg'``: Conjugate Gradient (requires symmetric positive
              definite system).
            - ``'gmres'``: Generalized Minimal Residual (general systems).
            - ``'bicgstab'``: BiConjugate Gradient Stabilized (general
              non-symmetric systems).
            - ``'lgmres'``: LGMRES variant of GMRES.
            - ``'minres'``: Minimal Residual (symmetric indefinite systems).
            - A callable with signature ``f(A, b, **kw) -> x`` or
              ``f(A, b, **kw) -> (x, info)``.  If the callable returns a
              2-tuple the second element is treated as a convergence flag
              (0 = success).

        **solver_options
            Additional keyword arguments passed to the iterative solver.
            Common options include:

            - ``rtol`` (float): Relative convergence tolerance.
            - ``maxiter`` (int): Maximum number of iterations.
            - ``x0`` (ndarray): Initial guess for the solution.
            - ``preconditioner`` (str or LinearOperator): If ``'ilu'``,
              an incomplete LU preconditioner is built automatically.
              Otherwise pass a ``scipy.sparse.linalg.LinearOperator``
              to use as the preconditioner ``M``.

        Returns
        -------
        out : numpy.ndarray
            Array with the solution of the PDE.

        Raises
        ------
        RuntimeError
            If an iterative solver fails to converge.
        ValueError
            If an unknown solver name is given or if solver_options are
            passed with the direct solver.
        """

        shape = self.bcs.shape
        if getattr(self.bcs, "is_time_dependent", False):
            raise ValueError(
                "Time-dependent boundary conditions are only supported by "
                "TimeDependentPDE."
            )
        if self._L is None:
            self._L = self.lhs.matrix(shape)  # expensive operation, so cache it

        L = sparse.lil_matrix(self._L)
        # Boundary values may require a wider dtype; keep the caller's rhs intact.
        f = np.array(
            self.rhs, dtype=np.result_type(self.rhs.dtype, self.bcs.rhs.dtype), copy=True
        ).reshape(-1, 1)

        nz = list(self.bcs.row_inds())

        L[nz, :] = self.bcs.lhs[nz, :]
        f[nz] = np.array(self.bcs.rhs[nz].toarray()).reshape(-1, 1)

        L = sparse.csr_matrix(L)

        # Direct solver (default, backward compatible)
        if solver is None or solver == 'direct':
            if solver_options:
                raise ValueError(
                    "solver_options are not supported with the direct solver"
                )
            return spsolve(L, f).reshape(shape)

        # Iterative solvers need f as a 1D array
        f_flat = np.asarray(f).ravel()

        # Handle preconditioner shorthand
        preconditioner = solver_options.pop('preconditioner', None)
        if preconditioner == 'ilu':
            from scipy.sparse.linalg import spilu, LinearOperator
            ilu = spilu(sparse.csc_matrix(L))
            solver_options['M'] = LinearOperator(L.shape, matvec=ilu.solve)
        elif preconditioner is not None:
            solver_options['M'] = preconditioner

        if callable(solver):
            result = solver(L, f_flat, **solver_options)
            if isinstance(result, tuple):
                x, info = result
                if info != 0:
                    raise RuntimeError(
                        f"Custom solver did not converge (info={info})"
                    )
            else:
                x = result
            return x.reshape(shape)

        if isinstance(solver, str):
            solve_func = self._get_iterative_solver(solver)
            x, info = solve_func(L, f_flat, **solver_options)
            if info > 0:
                raise RuntimeError(
                    f"Solver '{solver}' did not converge within "
                    f"the maximum number of iterations"
                )
            elif info < 0:
                raise RuntimeError(
                    f"Solver '{solver}' encountered an illegal input "
                    f"or breakdown (info={info})"
                )
            return x.reshape(shape)

        raise ValueError(
            f"solver must be None, 'direct', a solver name string, "
            f"or a callable, got {type(solver)}"
        )

    @staticmethod
    def _get_iterative_solver(name):
        """Resolve a solver name to a scipy function.

        Parameters
        ----------
        name : str

        Returns
        -------
        callable
        """
        from scipy.sparse.linalg import (
            cg, gmres, bicgstab, lgmres, minres,
        )
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


class BoundaryConditions:
    """
    Represents boundary conditions for a PDE.

    Supports Dirichlet, Neumann and Robin (mixed) boundary conditions.

    Dirichlet:
        ``bc[index] = value``

    Neumann:
        ``bc[index] = (diff_operator, value)``

    Robin (:math:`\\alpha u + \\beta \\partial u / \\partial n = g`):
        ``bc[index] = (alpha, diff_operator, beta, value)``

    For :class:`~findiff.TimeDependentPDE`, any ``value`` may also be a
    callable ``g(t)`` returning a scalar or an array (of the shape of the
    selected boundary points or of the whole grid), e.g.
    ``bc[0] = lambda t: np.sin(t)``.

    Alternatively, Robin BCs can be specified using the operator-tuple
    form ``bc[index] = (alpha * Identity() + beta * diff_operator, value)``.
    """

    def __init__(self, shape):
        """
        Initializes the BoundaryCondition object.

        The BoundaryCondition objects needs information about the
        grid on which to solve the PDE, specifically the shape
        of the (equidistant) grid.

        Parameters
        ----------
        shape: tuple of ints
            the number of grid points in each dimension

        """

        self.shape = shape
        siz = np.prod(shape)
        self.long_indices = np.array(list(range(siz))).reshape(shape)
        self.lhs = sparse.lil_matrix((siz, siz))
        self.rhs = sparse.lil_matrix((siz, 1))
        # time-dependent values: list of (flat row indices, callable g(t))
        self._time_dependent = []

    def __setitem__(self, key, value):
        """
        Sets the boundary condition for specific grid points.

        Parameters
        ----------
        key: int, tuple of ints or slices
            where (on what grid points) to apply the boundary condition.
            Specified by the indices (or slices) of the grid point(s).

        value:
            The boundary condition to apply.

            - **Dirichlet**: a scalar or array — ``bc[idx] = g``
            - **Neumann**: a 2-tuple ``(diff_op, g)`` — ``bc[idx] = (Diff(0, dx), g)``
            - **Robin**: a 4-tuple ``(alpha, diff_op, beta, g)`` representing
              :math:`\\alpha u + \\beta \\frac{\\partial u}{\\partial n} = g`.

            For time-dependent problems, ``g`` may be a callable ``g(t)``.
        """
        from findiff.operators import Identity

        lng_inds = self.long_indices[key]
        flat_inds = np.atleast_1d(lng_inds).reshape(-1)
        # a new assignment overrides earlier time-dependent values on these rows
        self._time_dependent = [
            (rows[~np.isin(rows, flat_inds)], func)
            for rows, func in self._time_dependent
        ]
        self._time_dependent = [(r, f) for r, f in self._time_dependent if len(r)]

        if isinstance(value, tuple) and len(value) == 4:
            # Robin BC: (alpha, diff_op, beta, rhs_value)
            alpha, diff_op, beta, rhs_value = value
            robin_op = alpha * Identity() + beta * diff_op
            mat = sparse.lil_matrix(robin_op.matrix(self.shape))
            self.lhs[lng_inds, :] = mat[lng_inds, :]
            value = rhs_value
        elif isinstance(value, tuple):
            # Neumann BC: (diff_op, rhs_value)
            op, value = value
            mat = sparse.lil_matrix(op.matrix(self.shape))
            self.lhs[lng_inds, :] = mat[lng_inds, :]
        else:
            # Dirichlet BC: scalar or array
            self.lhs[lng_inds, :] = 0
            self.lhs[lng_inds, lng_inds] = 1

        if callable(value):
            self._time_dependent.append((flat_inds, value))
            self.rhs[lng_inds] = 0
        elif isinstance(value, np.ndarray):
            value = value.reshape(-1)[lng_inds]
            for i, v in zip(lng_inds, value):
                self.rhs[i] = v
        else:
            self.rhs[lng_inds] = value

    def row_inds(self):
        nz_rows, nz_cols = self.lhs.nonzero()
        return nz_rows

    @property
    def is_time_dependent(self):
        """True if any boundary value is a callable of time."""
        return len(self._time_dependent) > 0

    def values_at(self, rows, t):
        """Returns the boundary values for the flat row indices `rows` at time `t`."""
        rows = np.asarray(rows, dtype=int)
        values = np.asarray(self.rhs.tocsr()[rows].toarray(), dtype=float).ravel()
        if not self._time_dependent:
            return values
        position = {row: i for i, row in enumerate(rows)}
        for td_rows, func in self._time_dependent:
            g = np.asarray(func(t), dtype=float)
            if g.size == 1:
                g = np.full(len(td_rows), float(g))
            elif g.size == len(td_rows):
                g = g.reshape(-1)
            elif g.shape == tuple(self.shape):
                g = g.reshape(-1)[td_rows]
            else:
                raise ValueError(
                    f"Time-dependent boundary value returned shape {g.shape}, "
                    f"expected a scalar, {len(td_rows)} values or the grid shape "
                    f"{tuple(self.shape)}."
                )
            for row, v in zip(td_rows, g):
                if row in position:
                    values[position[row]] = v
        return values
