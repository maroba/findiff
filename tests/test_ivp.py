"""Tests for time-dependent PDE solving (Method of Lines).

Covers:
- All four time-stepping methods
- 1D and 2D heat equation with exact solutions
- Convergence rate verification
- Periodic boundary conditions
- Callback functionality
- Iterative solver integration
- Solution storage (store_every)
- Error handling (invalid inputs)
- MOLSolution container interface
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from findiff import BoundaryConditions, Diff, TimeDependentPDE
from findiff.ivp import MOLSolution


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _make_1d_heat_problem(nx=101, nt=200, D=0.01, t_final=0.1):
    """1D heat equation with exact solution.

    u_t = D * u_xx on [0, 1]
    IC: u(x, 0) = sin(pi * x)
    BC: u(0) = u(1) = 0
    Exact: u(x, t) = sin(pi * x) * exp(-D * pi^2 * t)
    """
    x = np.linspace(0, 1, nx)
    dx = x[1] - x[0]
    L = D * Diff(0, dx) ** 2
    u0 = np.sin(np.pi * x)
    bc = BoundaryConditions((nx,))
    bc[0] = 0
    bc[-1] = 0
    t = np.linspace(0, t_final, nt)
    expected = np.sin(np.pi * x) * np.exp(-D * np.pi ** 2 * t_final)
    return TimeDependentPDE(L, u0, bc, t), expected


def _make_2d_heat_problem(nx=31, ny=31, nt=50, D=0.01, t_final=0.05):
    """2D heat equation with exact solution.

    u_t = D * (u_xx + u_yy) on [0, 1]^2
    IC: u(x, y, 0) = sin(pi * x) * sin(pi * y)
    BC: u = 0 on all edges
    Exact: u(x, y, t) = sin(pi*x) * sin(pi*y) * exp(-2*D*pi^2*t)
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing='ij')
    L = D * (Diff(0, dx) ** 2 + Diff(1, dy) ** 2)
    u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)
    bc = BoundaryConditions((nx, ny))
    bc[0, :] = 0
    bc[-1, :] = 0
    bc[:, 0] = 0
    bc[:, -1] = 0
    t = np.linspace(0, t_final, nt)
    expected = (np.sin(np.pi * X) * np.sin(np.pi * Y)
                * np.exp(-2 * D * np.pi ** 2 * t_final))
    return TimeDependentPDE(L, u0, bc, t), expected


# ---------------------------------------------------------------------------
# MOLSolution container
# ---------------------------------------------------------------------------

class TestMOLSolution:

    def test_attributes(self):
        t = np.array([0.0, 0.1, 0.2])
        u = np.random.rand(3, 10)
        sol = MOLSolution(t, u)
        assert sol.t.shape == (3,)
        assert sol.u.shape == (3, 10)

    def test_indexing(self):
        t = np.array([0.0, 0.1, 0.2])
        u = np.random.rand(3, 10)
        sol = MOLSolution(t, u)
        assert_array_almost_equal(sol[0], u[0])
        assert_array_almost_equal(sol[-1], u[-1])

    def test_final(self):
        t = np.array([0.0, 0.1, 0.2])
        u = np.random.rand(3, 10)
        sol = MOLSolution(t, u)
        assert_array_almost_equal(sol.final, u[-1])


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_invalid_method_raises(self):
        problem, _ = _make_1d_heat_problem()
        with pytest.raises(ValueError, match="Unknown method"):
            problem.solve(method='invalid')

    def test_t_must_be_1d(self):
        x = np.linspace(0, 1, 10)
        dx = x[1] - x[0]
        L = Diff(0, dx) ** 2
        bc = BoundaryConditions((10,))
        bc[0] = 0
        bc[-1] = 0
        with pytest.raises(ValueError, match="1D array"):
            TimeDependentPDE(L, np.zeros(10), bc, np.zeros((2, 3)))

    def test_t_too_short_raises(self):
        x = np.linspace(0, 1, 10)
        dx = x[1] - x[0]
        L = Diff(0, dx) ** 2
        bc = BoundaryConditions((10,))
        bc[0] = 0
        bc[-1] = 0
        with pytest.raises(ValueError, match="at least 2"):
            TimeDependentPDE(L, np.zeros(10), bc, np.array([0.0]))

    def test_u0_shape_mismatch_raises(self):
        x = np.linspace(0, 1, 10)
        dx = x[1] - x[0]
        L = Diff(0, dx) ** 2
        bc = BoundaryConditions((10,))
        bc[0] = 0
        bc[-1] = 0
        with pytest.raises(ValueError, match="does not match"):
            TimeDependentPDE(L, np.zeros(20), bc, np.linspace(0, 1, 5))

    def test_solver_options_rejected_for_explicit(self):
        problem, _ = _make_1d_heat_problem()
        with pytest.raises(ValueError, match="not used with explicit"):
            problem.solve(method='forward-euler', solver='gmres')


# ---------------------------------------------------------------------------
# Forward Euler
# ---------------------------------------------------------------------------

class TestForwardEuler1D:

    def test_heat_equation(self):
        # Forward Euler needs small dt for CFL stability
        problem, expected = _make_1d_heat_problem(
            nx=51, nt=5000, D=0.01, t_final=0.1
        )
        sol = problem.solve(method='forward-euler')
        assert_array_almost_equal(sol, expected, decimal=2)

    def test_returns_ndarray(self):
        problem, _ = _make_1d_heat_problem(nx=21, nt=500, D=0.01, t_final=0.01)
        result = problem.solve(method='forward-euler')
        assert isinstance(result, np.ndarray)
        assert result.shape == (21,)


# ---------------------------------------------------------------------------
# RK4
# ---------------------------------------------------------------------------

class TestRK4_1D:

    def test_heat_equation(self):
        problem, expected = _make_1d_heat_problem(
            nx=51, nt=2000, D=0.01, t_final=0.1
        )
        sol = problem.solve(method='rk4')
        assert_array_almost_equal(sol, expected, decimal=3)


# ---------------------------------------------------------------------------
# Backward Euler
# ---------------------------------------------------------------------------

class TestBackwardEuler1D:

    def test_heat_equation(self):
        problem, expected = _make_1d_heat_problem(nx=101, nt=200)
        sol = problem.solve(method='backward-euler')
        assert_array_almost_equal(sol, expected, decimal=2)


# ---------------------------------------------------------------------------
# Crank-Nicolson
# ---------------------------------------------------------------------------

class TestCrankNicolson1D:

    def test_heat_equation(self):
        problem, expected = _make_1d_heat_problem(nx=101, nt=200)
        sol = problem.solve(method='crank-nicolson')
        assert_array_almost_equal(sol, expected, decimal=3)


class TestCrankNicolson2D:

    def test_heat_equation(self):
        problem, expected = _make_2d_heat_problem()
        sol = problem.solve(method='crank-nicolson')
        assert_array_almost_equal(sol, expected, decimal=2)


# ---------------------------------------------------------------------------
# Iterative solvers
# ---------------------------------------------------------------------------

class TestImplicitWithIterativeSolver:

    def test_gmres_1d(self):
        problem, expected = _make_1d_heat_problem()
        sol = problem.solve(
            method='crank-nicolson',
            solver='gmres',
            preconditioner='ilu',
        )
        assert_array_almost_equal(sol, expected, decimal=3)

    def test_cg_1d(self):
        problem, expected = _make_1d_heat_problem()
        sol = problem.solve(
            method='backward-euler',
            solver='cg',
            preconditioner='ilu',
        )
        assert_array_almost_equal(sol, expected, decimal=2)


# ---------------------------------------------------------------------------
# Periodic BCs
# ---------------------------------------------------------------------------

class TestPeriodicBC:

    def test_advection_rk4(self):
        """Advection equation u_t = -c * u_x with periodic BC."""
        nx = 101
        c = 1.0
        x = np.linspace(0, 2 * np.pi, nx, endpoint=False)
        dx = x[1] - x[0]

        L = -c * Diff(0, dx, periodic=True)
        u0 = np.sin(x)

        # Minimal BCs (no actual boundary for periodic)
        bc = BoundaryConditions((nx,))

        t_final = 0.5
        nt = 500
        t = np.linspace(0, t_final, nt)

        problem = TimeDependentPDE(L, u0, bc, t)
        sol = problem.solve(method='rk4')

        expected = np.sin(x - c * t_final)
        assert_array_almost_equal(sol, expected, decimal=2)


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class TestCallback:

    def test_callback_called(self):
        problem, _ = _make_1d_heat_problem(nt=10)
        calls = []

        def cb(step, t, u):
            calls.append(step)

        problem.solve(method='crank-nicolson', callback=cb)
        assert calls == list(range(1, 10))

    def test_callback_early_stop(self):
        problem, _ = _make_1d_heat_problem(nt=100)

        def cb(step, t, u):
            if step >= 5:
                return False

        sol = problem.solve(method='crank-nicolson', callback=cb)
        # When store_every is None, early stop returns ndarray
        assert isinstance(sol, np.ndarray)

    def test_callback_early_stop_with_history(self):
        problem, _ = _make_1d_heat_problem(nt=100)

        def cb(step, t, u):
            if step >= 5:
                return False

        sol = problem.solve(
            method='crank-nicolson', callback=cb, store_every=1
        )
        assert isinstance(sol, MOLSolution)
        assert len(sol.t) <= 7  # 0..5 plus possibly the stop step


# ---------------------------------------------------------------------------
# store_every
# ---------------------------------------------------------------------------

class TestStoreEvery:

    def test_store_every_none_returns_ndarray(self):
        problem, _ = _make_1d_heat_problem(nx=21, nt=20)
        result = problem.solve(method='crank-nicolson')
        assert isinstance(result, np.ndarray)
        assert result.shape == (21,)

    def test_store_every_1_returns_full_history(self):
        problem, _ = _make_1d_heat_problem(nx=21, nt=20)
        sol = problem.solve(method='crank-nicolson', store_every=1)
        assert isinstance(sol, MOLSolution)
        assert sol.u.shape == (20, 21)
        assert sol.t.shape == (20,)

    def test_store_every_n(self):
        problem, _ = _make_1d_heat_problem(nx=21, nt=21)
        sol = problem.solve(method='crank-nicolson', store_every=5)
        assert isinstance(sol, MOLSolution)
        # Steps 0, 5, 10, 15, 20 = 5 snapshots (plus final if not aligned)
        assert len(sol.t) >= 5

    def test_initial_condition_preserved(self):
        problem, _ = _make_1d_heat_problem(nx=21, nt=20)
        sol = problem.solve(method='crank-nicolson', store_every=1)
        assert_array_almost_equal(sol[0], problem.u0)

    def test_final_property(self):
        problem, expected = _make_1d_heat_problem(nx=51, nt=100)
        sol = problem.solve(method='crank-nicolson', store_every=1)
        # .final should match the last stored snapshot
        assert_array_almost_equal(sol.final, sol[-1])


# ---------------------------------------------------------------------------
# Convergence rates
# ---------------------------------------------------------------------------

class TestConvergenceRates:
    """Verify temporal convergence order.

    Uses acc=4 spatial discretization so spatial error is negligible
    compared to temporal error, isolating the time-stepping order.
    """

    @staticmethod
    def _compute_errors(method, nt_values, nx, D, t_final, acc=4):
        x = np.linspace(0, 1, nx)
        dx = x[1] - x[0]
        L = D * Diff(0, dx, acc=acc) ** 2
        u0 = np.sin(np.pi * x)
        bc = BoundaryConditions((nx,))
        bc[0] = 0
        bc[-1] = 0
        exact = np.sin(np.pi * x) * np.exp(-D * np.pi ** 2 * t_final)

        errors = []
        dts = []
        for nt in nt_values:
            t = np.linspace(0, t_final, nt)
            problem = TimeDependentPDE(L, u0, bc, t)
            sol = problem.solve(method=method)
            errors.append(np.max(np.abs(sol - exact)))
            dts.append(t[1] - t[0])
        return np.array(dts), np.array(errors)

    def test_forward_euler_first_order(self):
        # CFL: dt < dx^2/(2D) = (1/50)^2/0.02 = 0.02, so nt>=500 is safe
        dts, errors = self._compute_errors(
            'forward-euler', [500, 1000, 2000],
            nx=51, D=0.01, t_final=0.1,
        )
        slope = np.polyfit(np.log(dts), np.log(errors), 1)[0]
        assert slope > 0.8, f"Forward Euler slope = {slope}, expected ~1.0"

    def test_crank_nicolson_second_order(self):
        dts, errors = self._compute_errors(
            'crank-nicolson', [20, 40, 80, 160],
            nx=101, D=0.1, t_final=0.5,
        )
        slope = np.polyfit(np.log(dts), np.log(errors), 1)[0]
        assert slope > 1.8, f"Crank-Nicolson slope = {slope}, expected ~2.0"

    def test_backward_euler_first_order(self):
        dts, errors = self._compute_errors(
            'backward-euler', [20, 40, 80, 160],
            nx=101, D=0.1, t_final=0.5,
        )
        slope = np.polyfit(np.log(dts), np.log(errors), 1)[0]
        assert slope > 0.8, f"Backward Euler slope = {slope}, expected ~1.0"
