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

    fd = Diff(dx, 1, 0)
    actual = fd(u)

    assert_array_almost_equal(expected, actual)


def test_partial_d2_dx2():
    shape = (101,)
    x, dx = make_grid(shape, (0, 1))

    u = x**2
    expected = 2

    fd = Diff(
        dx,
        2,
        0,
    )
    actual = fd(u, dx)

    assert_array_almost_equal(expected, actual)


def test_partial_d_dx_acc():
    shape = (11,)
    x, dx = make_grid(shape, (0, 1))

    u = x**3
    expected = 3 * x**2

    fd = Diff(dx, 1, 0)
    actual = fd(u, dx)
    np.testing.assert_raises(
        AssertionError, assert_array_almost_equal, expected, actual
    )

    actual = fd(u, acc=4)
    assert_array_almost_equal(expected, actual)


def test_partial_d2_dxdy():

    shape = 50, 50
    (x, y), (dx, dy), (X, Y) = make_grid(shape, ((0, 1), (0, 1)))
    u = np.sin(X) * np.sin(Y)

    fd = Diff(dx, 1, 0) * Diff(dy, 1, 1)
    actual = fd(u, h={0: dx, 1: dy})

    expected = np.cos(X) * np.cos(Y)
    assert_array_almost_equal(expected, actual, decimal=3)

    fd = Diff(dy, 1, 1) * Diff(dy, 1, 0)
    actual = fd(u)
    assert_array_almost_equal(expected, actual, decimal=3)


def test_partial_d_dx_on_2d_array():

    shape = 50, 50
    (x, y), (dx, dy), (X, Y) = make_grid(shape, ((0, 1), (0, 1)))
    u = np.sin(X) * np.sin(Y)

    fd = Diff(dy, 1, 1)
    actual = fd(u)

    expected = np.sin(X) * np.cos(Y)
    assert_array_almost_equal(expected, actual, decimal=3)


def test_add_two_diffs():

    shape = 50, 50
    (x, y), (dx, dy), (X, Y) = make_grid(shape, ((0, 1), (0, 1)))
    u = np.sin(X) * np.sin(Y)

    fd = Diff(dx, 2, 0) + Diff(dy, 2, 1)
    actual = fd(u)

    expected = -2 * np.sin(X) * np.sin(Y)
    assert_array_almost_equal(expected, actual, decimal=3)


def test_sub_two_diffs():

    shape = 50, 50
    (x, y), (dx, dy), (X, Y) = make_grid(shape, ((0, 1), (0, 1)))
    u = np.sin(X) * np.sin(Y)

    fd = Diff(dx, 2, 0) - Diff(dy, 2, 1)
    actual = fd(u)

    expected = np.zeros(shape)
    assert_array_almost_equal(expected, actual, decimal=3)


def test_multiply_with_scalar():

    shape = 50, 50
    (x, y), (dx, dy), (X, Y) = make_grid(shape, ((0, 1), (0, 1)))
    u = np.sin(X) * np.sin(Y)

    fd = 3 * Diff(dx, 2, 0)
    actual = fd(u, acc=4)

    expected = -3 * np.sin(X) * np.sin(Y)
    assert_array_almost_equal(expected, actual)


def test_multiply_with_variable_from_left():

    shape = 50, 50
    (x, y), (dx, dy), (X, Y) = make_grid(shape, ((0, 1), (0, 1)))
    u = np.sin(X) * np.sin(Y)

    fd = 3 * X * Y * Diff(dx, 2, 0)
    actual = fd(u, acc=4)

    expected = -3 * X * Y * np.sin(X) * np.sin(Y)
    assert_array_almost_equal(expected, actual)


def test_identity():

    shape = 50, 50
    (x, y), (dx, dy), (X, Y) = make_grid(shape, ((0, 1), (0, 1)))
    u = np.sin(X) * np.sin(Y)

    fd = Diff(dx, 2, 0) + Identity()
    actual = fd(u, acc=4)

    expected = np.zeros_like(u)
    assert_array_almost_equal(expected, actual)


def test_linear_comb():
    shape = 50, 50
    (x, y), (dx, dy), (X, Y) = make_grid(shape, ((0, 1), (0, 1)))
    u = np.sin(X) * np.sin(Y)

    fd = 3 * X * Diff(dx, 2, 0) - Y * (Diff(dy, 2, 1) + Diff(dx, 1, 0))
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
