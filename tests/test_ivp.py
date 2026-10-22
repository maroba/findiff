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
from scipy.linalg import expm

from findiff import BoundaryConditions, Diff, TimeDependentPDE
from findiff.ivp import MOLSolution


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method", ["forward-euler", "rk4", "backward-euler", "crank-nicolson"]
)
def test_initial_dirichlet_values(method):
    n = 11
    dx, diffusion, dt = 0.1, 0.1, 1e-5
    u0 = np.zeros(n)
    bc = BoundaryConditions(u0.shape)
    bc[0], bc[-1] = 1.0, 0.25
    problem = TimeDependentPDE(diffusion * Diff(0, dx) ** 2, u0, bc, [0, dt])

    solution = problem.solve(method=method, store_every=1)

    initial = u0.copy()
    initial[0], initial[-1] = 1.0, 0.25
    np.testing.assert_array_equal(solution.u[0], initial)
    # Exact evolution of the interior finite-difference heat equation.
    matrix = diffusion / dx**2 * (
        np.diag(np.full(n - 2, -2.0))
        + np.diag(np.ones(n - 3), 1)
        + np.diag(np.ones(n - 3), -1)
    )
    steady = np.linspace(1.0, 0.25, n)[1:-1]
    expected = steady + expm(dt * matrix) @ (initial[1:-1] - steady)
    np.testing.assert_allclose(solution.final[1:-1], expected, atol=2e-8)
    np.testing.assert_array_equal(u0, np.zeros(n))


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


def _heat_problem_2d(n=20):
    x = np.linspace(0, 1, n)
    dx = x[1] - x[0]
    X, Y = np.meshgrid(x, x, indexing="ij")
    L = 0.01 * (Diff(0, dx) ** 2 + Diff(1, dx) ** 2)
    bc = BoundaryConditions((n, n))
    bc[0, :] = 0
    bc[-1, :] = Diff(0, dx), 0.5
    bc[:, 0] = 0
    bc[:, -1] = 1
    return L, np.sin(np.pi * X) * np.sin(np.pi * Y), bc


@pytest.mark.parametrize("method", ["backward-euler", "crank-nicolson"])
def test_implicit_system_is_built_and_factorized_once_for_linspace_times(
    method, monkeypatch
):
    import findiff.ivp as ivp

    builds, factorizations = [], []
    original_build = ivp._ImplicitStepper._build_system
    original_splu = ivp.splu

    def counting_build(self, dt):
        builds.append(dt)
        return original_build(self, dt)

    def counting_splu(A):
        factorizations.append(A.shape)
        return original_splu(A)

    monkeypatch.setattr(ivp._ImplicitStepper, "_build_system", counting_build)
    monkeypatch.setattr(ivp, "splu", counting_splu)

    L, u0, bc = _heat_problem_2d()
    # np.linspace yields steps that differ by round-off only
    TimeDependentPDE(L, u0, bc, np.linspace(0, 0.1, 51)).solve(method=method)

    assert len(builds) == 1
    assert len(factorizations) == 1


@pytest.mark.parametrize("method", ["backward-euler", "crank-nicolson"])
def test_implicit_variable_time_step_matches_piecewise_solve(method):
    L, u0, bc = _heat_problem_2d()
    t1 = np.linspace(0, 0.05, 11)
    t2 = np.linspace(0.05, 0.1, 31)

    whole = TimeDependentPDE(L, u0, bc, np.concatenate([t1, t2[1:]])).solve(
        method=method
    )
    first = TimeDependentPDE(L, u0, bc, t1).solve(method=method)
    piecewise = TimeDependentPDE(L, first, bc, t2).solve(method=method)

    np.testing.assert_allclose(whole, piecewise, atol=1e-12)


@pytest.mark.parametrize("method", ["backward-euler", "crank-nicolson"])
def test_implicit_boundary_conditions_hold_every_step(method):
    L, u0, bc = _heat_problem_2d()
    x = np.linspace(0, 1, u0.shape[0])
    dx = x[1] - x[0]
    d_dx = Diff(0, dx)

    sol = TimeDependentPDE(L, u0, bc, np.linspace(0, 0.1, 21)).solve(
        method=method, store_every=1
    )

    for u in sol.u[1:]:
        np.testing.assert_allclose(u[0, :-1], 0, atol=1e-12)  # corner (0, -1) belongs to bc[:, -1]
        np.testing.assert_allclose(u[1:-1, -1], 1, atol=1e-12)
        np.testing.assert_allclose(d_dx(u)[-1, 1:-1], 0.5, atol=1e-8)


# u = sin(2t) + x^2 t solves u_t = u_xx + s with s = 2cos(2t) + x^2 - 2t;
# the spatial discretization is exact, so only the time error remains.
def _manufactured_error(method, nt, neumann=False):
    n = 11
    x = np.linspace(0, 1, n)
    dx = x[1] - x[0]
    bc = BoundaryConditions((n,))
    bc[0] = lambda t: np.sin(2 * t)
    if neumann:
        bc[-1] = Diff(0, dx), lambda t: 2 * t
    else:
        bc[-1] = lambda t: np.sin(2 * t) + t
    t = np.linspace(0, 0.5, nt + 1)
    u0 = np.sin(0) + x**2 * 0
    source = lambda t: 2 * np.cos(2 * t) + x**2 - 2 * t  # noqa: E731

    u = TimeDependentPDE(Diff(0, dx) ** 2, u0, bc, t, source=source).solve(
        method=method
    )
    return np.abs(u - (np.sin(1.0) + x**2 * 0.5)).max()


@pytest.mark.parametrize(
    "method, nt, order",
    [
        ("forward-euler", 400, 1),
        ("rk4", 200, 4),
        ("backward-euler", 40, 1),
        ("crank-nicolson", 40, 2),
    ],
)
def test_time_dependent_dirichlet_and_source_reach_method_order(method, nt, order):
    measured = np.log2(_manufactured_error(method, nt) / _manufactured_error(method, 2 * nt))
    assert measured == pytest.approx(order, abs=0.4)


@pytest.mark.parametrize("method, order", [("backward-euler", 1), ("crank-nicolson", 2)])
def test_time_dependent_neumann_reaches_method_order(method, order):
    e1 = _manufactured_error(method, 40, neumann=True)
    e2 = _manufactured_error(method, 80, neumann=True)
    assert np.log2(e1 / e2) == pytest.approx(order, abs=0.2)


def test_time_dependent_bc_value_shapes():
    shape = (6, 5)
    bc = BoundaryConditions(shape)
    bc[0, :] = lambda t: t                                  # scalar
    bc[-1, :] = lambda t: t * np.arange(5)                  # shape of the boundary slice
    bc[1:-1, 0] = lambda t: t * np.ones(shape)              # shape of the whole grid
    bc[1:-1, -1] = 7.0                                      # static
    rows = np.unique(np.fromiter(bc.row_inds(), dtype=int))

    values = dict(zip(rows, bc.values_at(rows, 2.0)))
    idx = np.arange(30).reshape(shape)

    assert all(values[i] == 2.0 for i in idx[0, :])
    assert [values[i] for i in idx[-1, :]] == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert all(values[i] == 2.0 for i in idx[1:-1, 0])
    assert all(values[i] == 7.0 for i in idx[1:-1, -1])
    assert bc.is_time_dependent

    bc[0, :] = lambda t: np.ones(3)
    with pytest.raises(ValueError, match="returned shape"):
        bc.values_at(rows, 0.0)


def test_static_assignment_overrides_time_dependent_value():
    bc = BoundaryConditions((5,))
    bc[0] = lambda t: t
    bc[0] = 3.0

    assert not bc.is_time_dependent
    assert bc.values_at([0], 10.0)[0] == 3.0


def test_constant_array_source():
    n = 21
    x = np.linspace(0, 1, n)
    dx = x[1] - x[0]
    bc = BoundaryConditions((n,))
    bc[0] = 0
    bc[-1] = 0
    # steady state of u_t = u_xx + 2 with u(0) = u(1) = 0 is u = x (1 - x)
    u = TimeDependentPDE(Diff(0, dx) ** 2, np.zeros(n), bc, np.linspace(0, 5, 201),
                         source=2.0).solve(method="backward-euler")

    np.testing.assert_allclose(u, x * (1 - x), atol=1e-6)


def test_pde_rejects_time_dependent_bcs():
    from findiff import PDE

    bc = BoundaryConditions((5,))
    bc[0] = lambda t: t
    bc[-1] = 0

    with pytest.raises(ValueError, match="only supported by TimeDependentPDE"):
        PDE(Diff(0, 0.1) ** 2, np.zeros(5), bc).solve()


# u = x^2 cos(t) solves u_tt = u_xx + s with s = -(x^2 + 2) cos(t); the spatial
# discretization is exact, so only the time error remains.
def _wave_manufactured_error(method, nt, neumann=False):
    n = 11
    x = np.linspace(0, 1, n)
    dx = x[1] - x[0]
    bc = BoundaryConditions((n,))
    bc[0] = 0
    if neumann:
        bc[-1] = Diff(0, dx), lambda t: 2 * np.cos(t)
    else:
        bc[-1] = lambda t: np.cos(t)
    source = lambda t: -(x**2 + 2) * np.cos(t)  # noqa: E731

    u = TimeDependentPDE(
        Diff(0, dx) ** 2, x**2, bc, np.linspace(0, 1, nt + 1),
        source=source, v0=np.zeros(n),
    ).solve(method=method)
    return np.abs(u - x**2 * np.cos(1.0)).max()


@pytest.mark.parametrize(
    "method, order",
    [("verlet", 2), ("rk4", 4), ("crank-nicolson", 2), ("backward-euler", 1)],
)
def test_second_order_in_time_reaches_method_order(method, order):
    e1 = _wave_manufactured_error(method, 50)
    e2 = _wave_manufactured_error(method, 100)
    assert np.log2(e1 / e2) == pytest.approx(order, abs=0.2)


@pytest.mark.parametrize("method, order", [("crank-nicolson", 2), ("backward-euler", 1)])
def test_second_order_in_time_with_time_dependent_neumann(method, order):
    e1 = _wave_manufactured_error(method, 50, neumann=True)
    e2 = _wave_manufactured_error(method, 100, neumann=True)
    assert np.log2(e1 / e2) == pytest.approx(order, abs=0.2)


def _standing_wave(method, periods=10):
    n = 101
    x = np.linspace(0, 1, n)
    dx = x[1] - x[0]
    bc = BoundaryConditions((n,))
    bc[0] = 0
    bc[-1] = 0
    dt = 0.5 * dx
    t = np.arange(0, 2 * periods + dt / 2, dt)
    return x, TimeDependentPDE(
        Diff(0, dx) ** 2, np.sin(np.pi * x), bc, t, v0=np.zeros(n)
    ).solve(method=method, store_every=len(t) - 1)


@pytest.mark.parametrize("method", ["verlet", "crank-nicolson"])
def test_standing_wave_keeps_its_amplitude(method):
    x, sol = _standing_wave(method)
    # after 10 periods, u = sin(pi x) and u_t = 0 again (up to phase error)
    np.testing.assert_allclose(sol.final, np.sin(np.pi * x), atol=2e-2)
    assert np.abs(sol.final).max() == pytest.approx(1, abs=1e-3)


def test_backward_euler_damps_waves():
    _, sol = _standing_wave("backward-euler")
    assert np.abs(sol.final).max() < 0.8  # about 0.61 after 10 periods


def test_second_order_solution_stores_velocity():
    n = 51
    x = np.linspace(0, 1, n)
    bc = BoundaryConditions((n,))
    bc[0] = 0
    bc[-1] = 0
    t = np.linspace(0, 0.5, 201)
    sol = TimeDependentPDE(
        Diff(0, x[1] - x[0]) ** 2, np.sin(np.pi * x), bc, t, v0=np.zeros(n)
    ).solve(method="verlet", store_every=100)

    assert sol.v.shape == sol.u.shape
    # u = sin(pi x) cos(pi t)  ->  u_t(0.5) = -pi sin(pi x)
    np.testing.assert_allclose(sol.v[-1], -np.pi * np.sin(np.pi * x), atol=1e-2)


def test_second_order_invalid_input():
    n = 11
    bc = BoundaryConditions((n,))
    bc[0] = 0
    bc[-1] = 0
    L = Diff(0, 0.1) ** 2

    with pytest.raises(ValueError, match="unstable"):
        TimeDependentPDE(L, np.zeros(n), bc, [0, 1], v0=np.zeros(n)).solve(
            method="forward-euler"
        )
    with pytest.raises(ValueError, match="requires a problem of second order"):
        TimeDependentPDE(L, np.zeros(n), bc, [0, 1]).solve(method="verlet")
    with pytest.raises(ValueError, match="v0 shape"):
        TimeDependentPDE(L, np.zeros(n), bc, [0, 1], v0=np.zeros(n + 1))
