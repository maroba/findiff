import numpy as np

from findiff import Diff, BoundaryConditions, PDE, Identity
from tests.test_diff import make_grid


def test_1d_dirichlet_hom():

    shape = (11,)

    x = np.linspace(0, 1, 11)
    dx = x[1] - x[0]
    L = Diff(0, dx) ** 2

    bc = BoundaryConditions(shape)

    bc[0] = 1
    bc[-1] = 2

    pde = PDE(L, np.zeros_like(x), bc)
    u = pde.solve()
    expected = x + 1
    np.testing.assert_array_almost_equal(expected, u)


def test_1d_dirichlet_periodic():

    shape = (101,)

    x = np.linspace(0, 2 * np.pi, 101, endpoint=False)
    dx = x[1] - x[0]
    L = Diff(0, dx, periodic=True) ** 2 + 1
    L.set_accuracy(4)

    bc = BoundaryConditions(shape)

    bc[0] = 1

    pde = PDE(L, np.zeros_like(x), bc)
    u = pde.solve()

    # import matplotlib.pyplot as plt
    # plt.plot(x, u)
    # plt.savefig("test.png")
    expected = np.cos(x)
    np.testing.assert_array_almost_equal(expected, u)


def test_1d_dirichlet_inhom():

    nx = 21
    shape = (nx,)

    x = np.linspace(0, 1, nx)
    dx = x[1] - x[0]
    L = Diff(0, dx) ** 2

    bc = BoundaryConditions(shape)

    bc[0] = 1
    bc[-1] = 2

    pde = PDE(L, 6 * x, bc)

    u = pde.solve()
    expected = x**3 + 1
    np.testing.assert_array_almost_equal(expected, u)


def test_1d_neumann_hom():
    nx = 11
    shape = (nx,)

    x = np.linspace(0, 1, nx)
    dx = x[1] - x[0]
    L = Diff(0, dx) ** 2

    bc = BoundaryConditions(shape)

    bc[0] = 1
    bc[-1] = Diff(0, dx), 2

    pde = PDE(L, np.zeros_like(x), bc)
    u = pde.solve()
    expected = 2 * x + 1
    np.testing.assert_array_almost_equal(expected, u)


def test_2d_dirichlet_hom():
    shape = (11, 11)

    x, y = np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1])
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")
    L = Diff(0, dx) ** 2 + Diff(1, dy) ** 2

    expected = X + 1

    bc = BoundaryConditions(shape)

    bc[0, :] = 1
    bc[-1, :] = 2
    bc[:, 0] = X + 1
    bc[:, -1] = X + 1

    pde = PDE(L, np.zeros_like(X), bc)
    u = pde.solve()

    np.testing.assert_array_almost_equal(expected, u)


def test_2d_dirichlet_periodic():
    shape = (101, 101)

    x, y = np.linspace(0, 2 * np.pi, shape[0], endpoint=False), np.linspace(
        0, 2 * np.pi, shape[1], endpoint=False
    )
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")
    L = Diff(0) ** 2 + 1
    L.set_accuracy(4)
    L.set_grid({0: {"h": dx, "periodic": True}, 1: {"h": dy, "periodic": True}})

    expected = np.cos(X) * np.sin(Y)

    bc = BoundaryConditions(shape)

    bc[0, :] = np.sin(Y)[0, :]
    # other boundary conditions are implicitly periodic through periodic grid

    bc[:, 0] = 0

    pde = PDE(L, np.zeros_like(X), bc)
    u = pde.solve()

    np.testing.assert_array_almost_equal(expected, u, decimal=6)


def test_2d_dirichlet_inhom():
    shape = (11, 11)

    x, y = np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1])
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")
    L = Diff(0, dx) ** 2 + Diff(1, dy) ** 2

    expected = X**3 + Y**3 + 1
    f = 6 * X + 6 * Y

    bc = BoundaryConditions(shape)

    bc[0, :] = expected
    bc[-1, :] = expected
    bc[:, 0] = expected
    bc[:, -1] = expected

    pde = PDE(L, f, bc)
    u = pde.solve()
    np.testing.assert_array_almost_equal(expected, u)


def test_2d_neumann_hom():
    shape = (31, 31)

    x, y = np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1])
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")
    L = Diff(0, dx) ** 2 + Diff(1, dy) ** 2

    expected = X**2 + Y**2 + 1
    f = 4 * np.ones_like(X)

    bc = BoundaryConditions(shape)

    d_dy = Diff(1, dy)

    bc[0, :] = expected
    bc[-1, :] = expected
    bc[:, 0] = d_dy, 2 * Y
    bc[:, -1] = d_dy, 2 * Y

    pde = PDE(L, f, bc)
    u = pde.solve()
    np.testing.assert_array_almost_equal(expected, u)


def test_1d_oscillator_free_dirichlet():
    n = 300
    shape = (n,)
    t = np.linspace(0, 5, n)
    dt = t[1] - t[0]
    L = Diff(0, dt) ** 2 + Identity()

    bc = BoundaryConditions(shape)

    bc[0] = 1
    bc[-1] = 2

    eq = PDE(L, np.zeros_like(t), bc)
    u = eq.solve()
    expected = np.cos(t) - (np.cos(5) - 2) * np.sin(t) / np.sin(5)
    np.testing.assert_array_almost_equal(expected, u, decimal=4)


def test_1d_damped_osc_driv_dirichlet():
    n = 100
    shape = (n,)
    t = np.linspace(0, 1, n)
    dt = t[1] - t[0]
    L = Diff(0, dt) ** 2 - Diff(0, dt) + Identity()
    f = -3 * np.exp(-t) * np.cos(t) + 2 * np.exp(-t) * np.sin(t)

    expected = np.exp(-t) * np.sin(t)

    bc = BoundaryConditions(shape)

    bc[0] = expected[0]
    bc[-1] = expected[-1]

    eq = PDE(L, f, bc)
    u = eq.solve()

    np.testing.assert_array_almost_equal(expected, u, decimal=4)


def test_1d_oscillator_driv_neumann():
    n = 200
    shape = (n,)
    t = np.linspace(0, 1, n)
    dt = t[1] - t[0]
    L = Diff(0, dt) ** 2 - Diff(0, dt) + Identity()
    f = -3 * np.exp(-t) * np.cos(t) + 2 * np.exp(-t) * np.sin(t)

    expected = np.exp(-t) * np.sin(t)

    bc = BoundaryConditions(shape)

    bc[0] = Diff(0, dt), 1
    bc[-1] = expected[-1]

    eq = PDE(L, f, bc)
    u = eq.solve()

    np.testing.assert_array_almost_equal(expected, u, decimal=4)


def test_1d_with_coeffs():
    n = 200
    shape = (n,)
    t = np.linspace(0, 1, n)
    dt = t[1] - t[0]
    L = t * Diff(0, dt) ** 2
    f = 6 * t**2

    bc = BoundaryConditions(shape)

    bc[0] = 0
    bc[-1] = 1

    eq = PDE(L, f, bc)
    u = eq.solve()
    expected = t**3

    np.testing.assert_array_almost_equal(expected, u, decimal=4)


def test_mixed_equation__with_coeffs_2d():
    shape = (41, 51)

    x, y = np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1])
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")
    L = Diff(0, dx) ** 2 + X * Y * Diff(0, dx) * Diff(1, dy) + Diff(1, dy) ** 2

    expected = X**3 + Y**3 + 1
    f = 6 * (X + Y)

    bc = BoundaryConditions(shape)

    bc[0, :] = expected
    bc[-1, :] = expected
    bc[:, 0] = expected
    bc[:, -1] = expected

    pde = PDE(L, f, bc)
    u = pde.solve()
    np.testing.assert_array_almost_equal(expected, u, decimal=4)


def test_2d_inhom_const_coefs_dirichlet_all():
    shape = (41, 50)
    (x, y), (dx, dy), (X, Y) = make_grid(shape, edges=[(-1, 1), (-1, 1)])

    expected = X**3 + Y**3 + X * Y + 1

    L = 3 * Diff(0, dx) ** 2 + 2 * Diff(0, dx) * Diff(1, dy) + Diff(1, dy) ** 2
    f = 2 + 18 * X + 6 * Y

    bc = BoundaryConditions(shape)
    bc[0, :] = expected
    bc[-1, :] = expected
    bc[:, 0] = expected
    bc[:, -1] = expected

    pde = PDE(L, f, bc)
    actual = pde.solve()
    np.testing.assert_array_almost_equal(expected, actual, decimal=4)


def test_2d_inhom_var_coefs_dirichlet_all():
    shape = (41, 50)
    (x, y), (dx, dy), (X, Y) = make_grid(shape, edges=[(-1, 1), (-1, 1)])

    expected = X**3 + Y**3 + X * Y + 1

    L = 3 * X * Diff(0, dx) ** 2 + 2 * Y * Diff(0, dx) * Diff(1, dy) + Diff(1, dy) ** 2
    f = 18 * X**2 + 8 * Y

    bc = BoundaryConditions(shape)
    bc[0, :] = expected
    bc[-1, :] = expected
    bc[:, 0] = expected
    bc[:, -1] = expected

    pde = PDE(L, f, bc)
    actual = pde.solve()
    np.testing.assert_array_almost_equal(expected, actual, decimal=4)
