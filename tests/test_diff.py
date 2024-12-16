import numpy as np

# import matplotlib.pyplot as plt
from numpy import cos, sin
from numpy.testing import assert_array_almost_equal

from findiff import Diff, Identity


def test_partial_d_dx():
    shape = (101,)
    x, dx = make_grid(shape, (0, 1))

    u = x**2
    expected = 2 * x

    fd = Diff(0, dx)
    actual = fd(u)

    assert_array_almost_equal(expected, actual)


def test_partial_d_dx_periodic():
    x = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    dx = x[1] - x[0]
    f = np.sin(x)
    expected = np.cos(x)
    actual = Diff(0, dx, periodic=True, acc=4)(f)

    assert_array_almost_equal(expected, actual)


def test_partial_d2_dx2_matrix_periodic():
    dx = 1
    expected = np.array(
        [
            [-2, 1, 0, 0, 0, 1],
            [1, -2, 1, 0, 0, 0],
            [0, 1, -2, 1, 0, 0],
            [0, 0, 1, -2, 1, 0],
            [0, 0, 0, 1, -2, 1],
            [1, 0, 0, 0, 1, -2],
        ],
        dtype=np.float64,
    )
    actual = (Diff(0, dx, periodic=True) ** 2).matrix((6,)).toarray()

    assert_array_almost_equal(expected, actual)


def test_partial_d2_dx2_matrix_periodic_2d():

    x = y = np.linspace(0, np.pi, 100, endpoint=False)
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")
    f = np.sin(X) ** 2 * np.sin(Y) ** 2

    d_dy = Diff(1, dy, periodic=True, acc=4)

    expected = 2 * np.sin(X) ** 2 * np.sin(Y) * np.cos(Y)

    actual = d_dy.matrix(f.shape).dot(f.reshape(-1)).reshape(100, 100)

    assert_array_almost_equal(expected, actual)


def test_diff_2d_periodic_in_one_axis():
    x = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    y = np.linspace(0, 1, 100)

    X, Y = np.meshgrid(x, y, indexing="ij")
    f = np.sin(X) ** 2 * np.sin(Y) ** 2

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    d_dx = Diff(0, acc=6)

    d_dx.set_grid({0: {"h": dx, "periodic": True}, 1: dy})
    expected = 2 * np.sin(X) * np.cos(X) * np.sin(Y) ** 2
    actual = d_dx(f)
    assert_array_almost_equal(expected, actual)


def test_partial_d2_dx2():
    shape = (101,)
    x, dx = make_grid(shape, (0, 1))

    u = x**2
    expected = 2

    fd = Diff(0, dx) ** 2

    actual = fd(u, dx)

    assert_array_almost_equal(expected, actual)


def test_partial_d_dx_acc():
    shape = (11,)
    x, dx = make_grid(shape, (0, 1))

    u = x**3
    expected = 3 * x**2

    fd = Diff(0, dx)
    actual = fd(u)
    np.testing.assert_raises(
        AssertionError, assert_array_almost_equal, expected, actual
    )

    actual = fd(u, acc=4)
    assert_array_almost_equal(expected, actual)


def test_partial_d2_dxdy():

    shape = 50, 50
    (x, y), (dx, dy), (X, Y) = make_grid(shape, ((0, 1), (0, 1)))
    u = np.sin(X) * np.sin(Y)

    fd = Diff(0) * Diff(1)
    fd.set_grid({0: dx, 1: dy})
    actual = fd(u)

    expected = np.cos(X) * np.cos(Y)
    assert_array_almost_equal(expected, actual, decimal=3)


def test_partial_d_dx_on_2d_array():

    shape = 50, 50
    (x, y), (dx, dy), (X, Y) = make_grid(shape, ((0, 1), (0, 1)))
    u = np.sin(X) * np.sin(Y)

    fd = Diff(1, dy)
    actual = fd(u)

    expected = np.sin(X) * np.cos(Y)
    assert_array_almost_equal(expected, actual, decimal=3)


def test_add_two_diffs():

    shape = 50, 50
    (x, y), (dx, dy), (X, Y) = make_grid(shape, ((0, 1), (0, 1)))
    u = np.sin(X) * np.sin(Y)

    fd = Diff(0, dx) ** 2 + Diff(1, dy) ** 2
    actual = fd(u)

    expected = -2 * np.sin(X) * np.sin(Y)
    assert_array_almost_equal(expected, actual, decimal=3)


def test_sub_two_diffs():

    shape = 50, 50
    (x, y), (dx, dy), (X, Y) = make_grid(shape, ((0, 1), (0, 1)))
    u = np.sin(X) * np.sin(Y)

    fd = Diff(0, dx) ** 2 - Diff(1, dy) ** 2
    actual = fd(u)

    expected = np.zeros(shape)
    assert_array_almost_equal(expected, actual, decimal=3)


def test_multiply_with_scalar():

    shape = 50, 50
    (x, y), (dx, dy), (X, Y) = make_grid(shape, ((0, 1), (0, 1)))
    u = np.sin(X) * np.sin(Y)

    fd = 3 * Diff(0, dx) ** 2
    actual = fd(u, acc=4)

    expected = -3 * np.sin(X) * np.sin(Y)
    assert_array_almost_equal(expected, actual)


def test_multiply_with_variable_from_left():

    shape = 50, 50
    (x, y), (dx, dy), (X, Y) = make_grid(shape, ((0, 1), (0, 1)))
    u = np.sin(X) * np.sin(Y)

    fd = 3 * X * Y * Diff(0, dx) ** 2
    actual = fd(u, acc=4)

    expected = -3 * X * Y * np.sin(X) * np.sin(Y)
    assert_array_almost_equal(expected, actual)


def test_identity():

    shape = 50, 50
    (x, y), (dx, dy), (X, Y) = make_grid(shape, ((0, 1), (0, 1)))
    u = np.sin(X) * np.sin(Y)

    fd = Diff(0, dx) ** 2 + Identity()
    actual = fd(u, acc=4)

    expected = np.zeros_like(u)
    assert_array_almost_equal(expected, actual)


def test_linear_comb():
    shape = 50, 50
    (x, y), (dx, dy), (X, Y) = make_grid(shape, ((0, 1), (0, 1)))
    u = np.sin(X) * np.sin(Y)

    fd = 3 * X * Diff(0, dx) ** 2 - Y * (Diff(1, dy) ** 2 + Diff(0, dx))
    actual = fd(u, acc=4)

    expected = -3 * X * u - Y * (-sin(X) * sin(Y) + cos(X) * sin(Y))
    assert_array_almost_equal(expected, actual)


def make_grid(shape, edges):

    if len(shape) == 1:
        x = np.linspace(*edges, shape[0])
        dx = x[1] - x[0]
        return x, dx

    axes = tuple(
        [np.linspace(edges[k][0], edges[k][1], shape[k]) for k in range(len(shape))]
    )
    coords = np.meshgrid(*axes, indexing="ij")
    spacings = [axes[k][1] - axes[k][0] for k in range(len(shape))]
    return axes, spacings, coords


def test_nonuniform_1d_different_accs():
    x = np.r_[np.arange(0, 4, 0.05), np.arange(4, 10, 1)]
    f = np.exp(-(x**2))

    d_dx = Diff(0, x, acc=4)
    f_x = d_dx(f)
    assert_array_almost_equal(-2 * x * np.exp(-(x**2)), f_x, decimal=4)

    # same, but this time with default acc
    x = np.linspace(0, 1, 100)
    f = np.exp(-(x**2))
    d_dx = Diff(0, x)
    f_x = d_dx(f)
    assert_array_almost_equal(-2 * x * np.exp(-(x**2)), f_x, decimal=4)


def test_3d_different_accs():
    x = np.r_[np.arange(0, 4, 0.05), np.arange(4, 10, 1)]
    y = np.r_[np.arange(0, 4, 0.05), np.arange(4, 10, 1)]
    z = np.r_[np.arange(0, 4.5, 0.05), np.arange(4.5, 10, 1)]
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    f = np.exp(-(X**2) - Y**2 - Z**2)

    d_dy = Diff(1, y, acc=4)
    fy = d_dy(f)
    fye = -2 * Y * np.exp(-(X**2) - Y**2 - Z**2)
    assert_array_almost_equal(fy, fye, decimal=4)

    d_dy = Diff(1, y, acc=6)
    fy = d_dy(f)
    assert_array_almost_equal(fy, fye, decimal=4)
