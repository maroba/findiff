"""Stability of discrete second derivatives in time-dependent problems.

With Dirichlet conditions, the discrete second derivative must have eigenvalues
with negative real part for diffusion problems, and real negative eigenvalues
for wave equations; otherwise the solution grows exponentially regardless of
the time-stepping method.
"""

import warnings

import numpy as np
import pytest

from findiff import BoundaryConditions, CompactScheme, Diff, TimeDependentPDE


def _spectrum(d, n):
    return np.linalg.eigvals(d.matrix((n,)).toarray()[1:-1, 1:-1])


def _chebyshev(n):
    return (1 - np.cos(np.pi * np.linspace(0, 1, n))) / 2


def _tanh(n):
    return 1 + np.tanh(2.5 * (np.linspace(0, 1, n) - 1)) / np.tanh(2.5)


@pytest.mark.parametrize("acc", [2, 4, 6, 8])
def test_centered_boundary_gives_real_negative_spectrum(acc):
    n = 101
    ev = _spectrum(Diff(0, 1 / (n - 1), acc=acc, boundary="centered") ** 2, n)
    assert ev.real.max() < 0
    assert np.abs(ev.imag).max() < 1e-6 * np.abs(ev).max()


@pytest.mark.parametrize("acc", [4, 6])
def test_centered_boundary_wave_equation_is_stable_and_accurate(acc):
    def error(n):
        x = np.linspace(0, 1, n)
        bc = BoundaryConditions((n,))
        bc[0] = 0
        bc[-1] = 0
        u0 = np.sin(np.pi * x) + 0.4 * np.sin(2 * np.pi * x)
        t = np.linspace(0, 0.5, 2001)
        u = TimeDependentPDE(
            Diff(0, x[1] - x[0], acc=acc, boundary="centered") ** 2, u0, bc, t,
            v0=np.zeros(n),
        ).solve(method="rk4")
        exact = np.sin(np.pi * x) * np.cos(np.pi * 0.5) + 0.4 * np.sin(2 * np.pi * x) * np.cos(np.pi)
        return np.abs(u - exact).max()

    e1, e2 = error(41), error(81)
    assert e2 < 1e-4
    assert np.log2(e1 / e2) > 3.5


@pytest.mark.parametrize("order", [1, 2, 3])
def test_centered_boundary_matrix_matches_call(order):
    x = np.linspace(0, 1, 30)
    y = np.linspace(0, 2, 25)
    X, Y = np.meshgrid(x, y, indexing="ij")
    f = np.sin(2 * X) * np.cos(Y)
    for dim, h in ((0, x[1] - x[0]), (1, y[1] - y[0])):
        d = Diff(dim, h, acc=6, boundary="centered") ** order
        expected = d(f)
        actual = d.matrix(f.shape).dot(f.reshape(-1)).reshape(f.shape)
        np.testing.assert_allclose(actual, expected, atol=1e-8 * np.abs(expected).max())


def test_centered_boundary_invalid_use():
    with pytest.raises(ValueError, match="Unknown boundary closure"):
        Diff(0, 0.1, boundary="foo")(np.ones(10))
    with pytest.raises(ValueError, match="not supported for compact"):
        Diff(0, 0.1, compact=3, boundary="centered")(np.ones(10))
    with pytest.raises(NotImplementedError, match="equidistant"):
        Diff(0, np.linspace(0, 1, 10) ** 2, boundary="centered")(np.ones(10))


@pytest.mark.parametrize("grid", [_chebyshev, _tanh])
@pytest.mark.parametrize("acc", [2, 4])
def test_nonuniform_second_derivative_is_stable(grid, acc):
    n = 101
    ev = _spectrum(Diff(0, grid(n), acc=acc) ** 2, n)
    assert ev.real.max() < 0
    assert np.abs(ev.imag).max() < 1e-6 * np.abs(ev).max()


def test_nonuniform_boundary_rows_include_boundary_point():
    x = _chebyshev(30)
    for acc in (2, 4, 6):
        m = (Diff(0, x, acc=acc) ** 2).matrix((30,)).toarray()
        num_side = (acc + 2) // 2
        for i in range(1, num_side):
            assert m[i, 0] != 0
            assert m[-1 - i, -1] != 0


def _warns(lhs, n, second_order=False):
    bc = BoundaryConditions((n,))
    bc[0] = 0
    bc[-1] = 0
    v0 = np.zeros(n) if second_order else None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TimeDependentPDE(lhs, np.zeros(n), bc, [0, 1], v0=v0)
    return any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_stability_warning():
    n, dx = 101, 0.01
    custom = Diff(0, dx)
    custom.set_scheme(CompactScheme(2, {-1: 2 / 11, 0: 1, 1: 2 / 11}, [-2, -1, 0, 1, 2]))
    custom._order = 2

    # unstable
    assert _warns(Diff(0, dx, acc=6) ** 2, n)
    assert _warns(4.0 * Diff(0, dx, acc=4) ** 2, n, second_order=True)
    assert _warns(Diff(0, dx, compact=3, acc=6) ** 2, n, second_order=True)
    assert _warns(custom, n, second_order=True)
    # stable
    assert not _warns(Diff(0, dx, acc=4) ** 2, n)
    assert not _warns(Diff(0, dx) ** 2, n, second_order=True)
    assert not _warns(Diff(0, dx, acc=6, boundary="centered") ** 2, n)
    assert not _warns(Diff(0, dx, acc=8, boundary="centered") ** 2, n, second_order=True)
    assert not _warns(Diff(0, dx, acc=6, periodic=True) ** 2, n)
    assert not _warns(Diff(0, dx, compact=3, acc=4) ** 2, n, second_order=True)
    assert not _warns(Diff(0, _chebyshev(n), acc=4) ** 2, n, second_order=True)
