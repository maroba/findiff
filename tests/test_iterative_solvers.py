"""Tests for iterative solver support in PDE.solve().

Covers:
- Backward compatibility (default solver unchanged)
- String solver names (cg, gmres, bicgstab, lgmres, minres)
- Custom callable solver
- Preconditioner support (ILU shorthand and custom LinearOperator)
- solver_options passthrough (rtol, maxiter, x0)
- Convergence failure handling
- Error handling for invalid inputs
- 1D and 2D problems with different BC types
- Solver reuse with cached matrix
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.sparse.linalg import spsolve, LinearOperator

from findiff import Diff, PDE, BoundaryConditions, Identity


def _make_1d_laplacian_problem(n=101):
    """Build a 1D Laplacian BVP: u'' = 0, u(0) = 1, u(1) = 2."""
    x = np.linspace(0, 1, n)
    dx = x[1] - x[0]
    L = Diff(0, dx) ** 2
    bc = BoundaryConditions((n,))
    bc[0] = 1
    bc[-1] = 2
    pde = PDE(L, np.zeros_like(x), bc)
    expected = x + 1
    return pde, expected


def _make_2d_laplacian_problem(shape=(21, 21)):
    """Build a 2D Laplacian BVP: nabla^2 u = 0, u = X + 1 on boundary."""
    x = np.linspace(0, 1, shape[0])
    y = np.linspace(0, 1, shape[1])
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing='ij')
    L = Diff(0, dx) ** 2 + Diff(1, dy) ** 2
    expected = X + 1
    bc = BoundaryConditions(shape)
    # BoundaryConditions expects full grid arrays (not slices) for array values
    bc[0, :] = 1
    bc[-1, :] = 2
    bc[:, 0] = expected
    bc[:, -1] = expected
    pde = PDE(L, np.zeros_like(X), bc)
    return pde, expected


class TestBackwardCompatibility:

    def test_default_solver_matches_original(self):
        pde, expected = _make_1d_laplacian_problem()
        u = pde.solve()
        assert_array_almost_equal(u, expected)

    def test_explicit_direct_solver(self):
        pde, expected = _make_1d_laplacian_problem()
        u = pde.solve(solver='direct')
        assert_array_almost_equal(u, expected)

    def test_direct_solver_rejects_options(self):
        pde, _ = _make_1d_laplacian_problem()
        with pytest.raises(ValueError, match="solver_options are not supported"):
            pde.solve(solver='direct', rtol=1e-8)

    def test_none_solver_rejects_options(self):
        pde, _ = _make_1d_laplacian_problem()
        with pytest.raises(ValueError, match="solver_options are not supported"):
            pde.solve(rtol=1e-8)


class TestIterativeSolvers1D:
    """Test solver names on a 1D Dirichlet problem with ILU preconditioning.

    The finite-difference Laplacian produces a poorly-scaled matrix after
    BC injection (interior rows ~ 1/dx^2, boundary rows ~ 1), so
    preconditioning is needed for reliable iterative convergence.
    """

    @pytest.mark.parametrize('solver', ['cg', 'gmres', 'bicgstab', 'lgmres'])
    def test_solvers_with_ilu_1d_dirichlet(self, solver):
        pde, expected = _make_1d_laplacian_problem()
        u = pde.solve(solver=solver, preconditioner='ilu')
        assert_array_almost_equal(u, expected, decimal=4)

    def test_minres_available(self):
        """minres is available as a solver name (requires symmetric system)."""
        pde, expected = _make_1d_laplacian_problem()
        # minres requires a symmetric matrix; after BC injection, the
        # system is typically not symmetric. Just verify the dispatch
        # resolves without error by providing the exact solution as x0.
        u = pde.solve(solver='minres', x0=expected, maxiter=1)
        # With exact x0 and 1 iteration, result should be very close
        assert_array_almost_equal(u, expected, decimal=2)

    def test_1d_inhomogeneous(self):
        """u'' = 6x, u(0) = 1, u(1) = 2 => u = x^3 + 1."""
        n = 101
        x = np.linspace(0, 1, n)
        dx = x[1] - x[0]
        L = Diff(0, dx) ** 2
        bc = BoundaryConditions((n,))
        bc[0] = 1
        bc[-1] = 2
        pde = PDE(L, 6 * x, bc)
        u = pde.solve(solver='gmres', preconditioner='ilu')
        expected = x ** 3 + 1
        assert_array_almost_equal(u, expected, decimal=4)


class TestIterativeSolvers2D:

    def test_gmres_2d_dirichlet(self):
        pde, expected = _make_2d_laplacian_problem()
        u = pde.solve(solver='gmres', preconditioner='ilu', maxiter=500)
        assert_array_almost_equal(u, expected, decimal=4)

    def test_bicgstab_2d_dirichlet(self):
        pde, expected = _make_2d_laplacian_problem()
        u = pde.solve(solver='bicgstab', preconditioner='ilu', maxiter=500)
        assert_array_almost_equal(u, expected, decimal=4)

    def test_bicgstab_2d_inhomogeneous(self):
        """nabla^2 u = 6X + 6Y, u = X^3 + Y^3 + 1 on boundary."""
        shape = (21, 21)
        x = np.linspace(0, 1, shape[0])
        y = np.linspace(0, 1, shape[1])
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing='ij')
        L = Diff(0, dx) ** 2 + Diff(1, dy) ** 2
        expected = X ** 3 + Y ** 3 + 1
        f = 6 * X + 6 * Y
        bc = BoundaryConditions(shape)
        bc[0, :] = expected
        bc[-1, :] = expected
        bc[:, 0] = expected
        bc[:, -1] = expected
        pde = PDE(L, f, bc)
        u = pde.solve(solver='bicgstab', preconditioner='ilu', maxiter=500)
        assert_array_almost_equal(u, expected, decimal=4)


class TestIterativeSolversWithNeumann:

    def test_gmres_1d_neumann(self):
        """u'' = 0, u(0) = 1, u'(1) = 2 => u = 2x + 1."""
        n = 101
        x = np.linspace(0, 1, n)
        dx = x[1] - x[0]
        L = Diff(0, dx) ** 2
        bc = BoundaryConditions((n,))
        bc[0] = 1
        bc[-1] = Diff(0, dx), 2
        pde = PDE(L, np.zeros_like(x), bc)
        u = pde.solve(solver='gmres', preconditioner='ilu')
        expected = 2 * x + 1
        assert_array_almost_equal(u, expected, decimal=4)

    def test_bicgstab_2d_neumann(self):
        """nabla^2 u = 4, Dirichlet on x edges, Neumann on y edges."""
        shape = (21, 21)
        x = np.linspace(0, 1, shape[0])
        y = np.linspace(0, 1, shape[1])
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing='ij')
        L = Diff(0, dx) ** 2 + Diff(1, dy) ** 2
        expected = X ** 2 + Y ** 2 + 1
        f = 4 * np.ones_like(X)
        bc = BoundaryConditions(shape)
        bc[0, :] = expected
        bc[-1, :] = expected
        # Neumann BCs: pass full grid array for the value
        bc[:, 0] = Diff(1, dy), 2 * Y
        bc[:, -1] = Diff(1, dy), 2 * Y
        pde = PDE(L, f, bc)
        u = pde.solve(solver='bicgstab', preconditioner='ilu', maxiter=2000)
        assert_array_almost_equal(u, expected, decimal=3)


class TestPreconditioner:

    def test_ilu_preconditioner_1d(self):
        pde, expected = _make_1d_laplacian_problem()
        u = pde.solve(solver='gmres', preconditioner='ilu')
        assert_array_almost_equal(u, expected, decimal=4)

    def test_ilu_preconditioner_2d(self):
        pde, expected = _make_2d_laplacian_problem(shape=(21, 21))
        u = pde.solve(solver='gmres', preconditioner='ilu', maxiter=500)
        assert_array_almost_equal(u, expected, decimal=4)

    def test_custom_linear_operator_preconditioner(self):
        """Diagonal (Jacobi) preconditioner as a LinearOperator."""
        pde, expected = _make_1d_laplacian_problem()

        # Build the system matrix to extract diagonal
        import scipy.sparse as sparse
        shape = pde.bcs.shape
        L_mat = pde.lhs.matrix(shape)
        L_lil = sparse.lil_matrix(L_mat)
        nz = list(pde.bcs.row_inds())
        L_lil[nz, :] = pde.bcs.lhs[nz, :]
        L_csr = sparse.csr_matrix(L_lil)
        diag = L_csr.diagonal()
        diag[diag == 0] = 1.0  # avoid division by zero
        M = LinearOperator(L_csr.shape, matvec=lambda x: x / diag)

        u = pde.solve(solver='gmres', preconditioner=M)
        assert_array_almost_equal(u, expected, decimal=4)

    def test_ilu_matches_manual_construction(self):
        """ILU shorthand should give same result as manual ILU."""
        import scipy.sparse as sparse
        from scipy.sparse.linalg import spilu

        pde, _ = _make_1d_laplacian_problem()

        u_shorthand = pde.solve(solver='gmres', preconditioner='ilu')

        # Manually construct ILU
        shape = pde.bcs.shape
        L_mat = pde.lhs.matrix(shape)
        L_lil = sparse.lil_matrix(L_mat)
        nz = list(pde.bcs.row_inds())
        L_lil[nz, :] = pde.bcs.lhs[nz, :]
        L_csr = sparse.csr_matrix(L_lil)
        ilu = spilu(sparse.csc_matrix(L_csr))
        M = LinearOperator(L_csr.shape, matvec=ilu.solve)

        u_manual = pde.solve(solver='gmres', M=M)
        assert_array_almost_equal(u_shorthand, u_manual)


class TestSolverOptions:

    def test_rtol_passthrough(self):
        pde, expected = _make_1d_laplacian_problem()
        u = pde.solve(solver='gmres', preconditioner='ilu', rtol=1e-8)
        assert_array_almost_equal(u, expected, decimal=6)

    def test_maxiter_triggers_failure(self):
        pde, _ = _make_2d_laplacian_problem(shape=(21, 21))
        with pytest.raises(RuntimeError, match="did not converge"):
            pde.solve(solver='lgmres', maxiter=1)

    def test_x0_initial_guess(self):
        pde, expected = _make_1d_laplacian_problem()
        # Provide exact solution as initial guess — should converge immediately
        u = pde.solve(solver='gmres', x0=expected, preconditioner='ilu')
        assert_array_almost_equal(u, expected, decimal=6)


class TestCustomCallable:

    def test_callable_returning_array(self):
        pde, expected = _make_1d_laplacian_problem()
        u = pde.solve(solver=lambda A, b: spsolve(A, b))
        assert_array_almost_equal(u, expected)

    def test_callable_returning_tuple_success(self):
        pde, expected = _make_1d_laplacian_problem()
        u = pde.solve(solver=lambda A, b: (spsolve(A, b), 0))
        assert_array_almost_equal(u, expected)

    def test_callable_returning_tuple_failure(self):
        pde, expected = _make_1d_laplacian_problem()
        with pytest.raises(RuntimeError, match="Custom solver did not converge"):
            pde.solve(solver=lambda A, b: (np.zeros(b.shape), 1))

    def test_callable_receives_solver_options(self):
        pde, expected = _make_1d_laplacian_problem()
        received = {}

        def my_solver(A, b, **kwargs):
            received.update(kwargs)
            return spsolve(A, b)

        pde.solve(solver=my_solver, rtol=1e-8, maxiter=100)
        assert received['rtol'] == 1e-8
        assert received['maxiter'] == 100


class TestErrorHandling:

    def test_unknown_solver_raises_value_error(self):
        pde, _ = _make_1d_laplacian_problem()
        with pytest.raises(ValueError, match="Unknown solver 'nonexistent'"):
            pde.solve(solver='nonexistent')

    def test_invalid_solver_type_raises_value_error(self):
        pde, _ = _make_1d_laplacian_problem()
        with pytest.raises(ValueError, match="solver must be"):
            pde.solve(solver=42)

    def test_nonconvergence_raises_runtime_error(self):
        pde, _ = _make_2d_laplacian_problem(shape=(21, 21))
        with pytest.raises(RuntimeError):
            pde.solve(solver='lgmres', maxiter=1)


class TestSolverReuse:

    def test_same_pde_different_solvers(self):
        pde, expected = _make_1d_laplacian_problem()
        u_direct = pde.solve(solver='direct')
        u_gmres = pde.solve(solver='gmres', preconditioner='ilu')
        assert_array_almost_equal(u_direct, u_gmres, decimal=4)

    def test_matrix_cached_across_solves(self):
        pde, _ = _make_1d_laplacian_problem()
        pde.solve()  # first call caches _L
        assert pde._L is not None
        cached = pde._L
        pde.solve(solver='gmres', preconditioner='ilu')  # reuses cache
        assert pde._L is cached
