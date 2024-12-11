import numpy as np
from numpy.testing import assert_array_almost_equal

from findiff.operators import Add, FieldOperator
from findiff import Diff, Identity


def test_applying_identity():

    f = np.array([1, 2, 3])
    I = Identity()

    assert np.array_equal(I(f), f)


def test_add_identities():

    f = np.array([1, 2, 3])
    I = Identity()

    assert np.array_equal((I + I + I)(f), 3 * f)


def test_add_scalar_to_identity():

    f = np.array([1, 2, 3])
    I = Identity()

    Ip1 = I + 2

    assert isinstance(Ip1, Add)
    assert isinstance(Ip1.left, Identity)
    assert isinstance(Ip1.right, FieldOperator)
    assert np.array_equal(Ip1(f), 3 * f)

    Ip1 = 2 + I

    assert isinstance(Ip1, Add)
    assert isinstance(Ip1.left, Identity)
    assert isinstance(Ip1.right, FieldOperator)
    assert np.array_equal(Ip1(f), 3 * f)


def test_add_scalars():

    f = np.array([1, 2, 3])
    S = FieldOperator(2)

    assert np.array_equal((S + S)(f), 4 * f)


def test_sub_scalars():

    f = np.array([1, 2, 3])
    S = FieldOperator(2)

    actual = (S - 2 * S)(f)
    expected = -2 * f
    np.testing.assert_array_almost_equal(actual, expected)


def test_multiply_scalars():

    f = np.array([1, 2, 3])
    S = FieldOperator(3)

    assert np.array_equal((S * S)(f), 9 * f)


def test_multiply_scalar_with_number():

    f = np.array([1, 2, 3])
    S = FieldOperator(3)

    assert np.array_equal((S * 2)(f), 6 * f)
    assert np.array_equal((2 * S)(f), 6 * f)


def test_multiply_scalar_with_field():

    f = np.array([1, 2, 3])
    g = np.array([-1, 1, -1])
    S = FieldOperator(3)

    assert np.array_equal((S * g)(f), [-3, 6, -9])
    assert np.array_equal((g * S)(f), [-3, 6, -9])


def test_mix_addition_and_multiplication():
    f = np.array([1, 2, 3])
    S2 = FieldOperator(2)
    S3 = FieldOperator(3)

    assert np.array_equal((S2 + S3 * 4)(f), 2 * f + 3 * 4 * f)


def test_simple_derivatives():

    x = np.linspace(0, 1, 100)
    dx = x[1] - x[0]
    f = x**3

    d_dx = Diff(0, dx)

    actual = d_dx(f)
    np.testing.assert_array_almost_equal(actual, 3 * x**2, decimal=3)

    actual = d_dx(f, acc=4)
    np.testing.assert_array_almost_equal(actual, 3 * x**2)

    d_dx = Diff(0, dx)

    actual = d_dx(f, acc=4)
    np.testing.assert_array_almost_equal(actual, 3 * x**2)

    d2_dx2 = Diff(0, dx) ** 2

    actual = d2_dx2(f, acc=4)
    np.testing.assert_array_almost_equal(actual, 6 * x)


def test_pow_diff():

    d = Diff(0)
    assert d.order == 1

    d2 = d**2

    assert d2.order == 2
    assert d.order == 1


def test_chained_derivatives():

    x = np.linspace(0, 1, 100)
    dx = x[1] - x[0]
    f = x**3

    d_dx = Diff(0, dx)

    d2_dx2 = d_dx * d_dx
    actual = d2_dx2(f, acc=4)
    np.testing.assert_array_almost_equal(actual, 6 * x)


def test_set_grid_lazily():

    x = np.linspace(0, 1, 100)
    dx = x[1] - x[0]
    f = x**3

    # no grid defined here:
    d_dx = Diff(0)

    diff_op = d_dx * d_dx + d_dx

    # now set grid for complete differential operator
    diff_op.set_grid({0: dx})

    actual = diff_op(f, acc=4)
    np.testing.assert_array_almost_equal(actual, 6 * x + 3 * x**2)


def test_harmonic():

    x = np.linspace(0, 1, 100)
    dx = x[1] - x[0]
    f = x**3

    T = -0.5 * Diff(0, dx) ** 2
    V = 0.5 * x**2
    H = T + V

    actual = H(f, acc=4)
    Tf = T(f, acc=4)
    Vf = FieldOperator(V)(f)
    np.testing.assert_array_almost_equal(Tf, -3 * x)
    np.testing.assert_array_almost_equal(Vf, 0.5 * x**2 * f)
    np.testing.assert_array_almost_equal(actual, -3 * x + 0.5 * x**2 * f)


def test_partial_diff():
    nx = 100
    x = np.linspace(0, np.pi, nx)
    u = np.sin(x)
    ux_ex = np.cos(x)
    dx = x[1] - x[0]
    fd = Diff(0, dx, acc=4)

    ux = fd(u)

    assert_array_almost_equal(ux, ux_ex, decimal=5)

    ny = 80
    y = np.linspace(0, np.pi, ny)
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    u = np.sin(X) * np.sin(Y)
    uxy_ex = np.cos(X) * np.cos(Y)

    # mixed partial derivative d2_dxdy
    fd = Diff(0, dx) * Diff(1, dy)

    uxy = fd(u, acc=4)

    assert_array_almost_equal(uxy, uxy_ex, decimal=5)


def test_matrix_1d():

    x = np.linspace(0, 6, 7)
    d2_dx2 = Diff(0, x[1] - x[0]) ** 2
    u = x**2

    mat = d2_dx2.matrix(u.shape)

    np.testing.assert_array_almost_equal(2 * np.ones_like(x), mat.dot(u.reshape(-1)))


def test_matrix_2d():
    thr = np.get_printoptions()["threshold"]
    lw = np.get_printoptions()["linewidth"]
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=500)
    x, y = [np.linspace(0, 4, 5)] * 2
    X, Y = np.meshgrid(x, y, indexing="ij")
    laplace = Diff(0, x[1] - x[0]) ** 2 + Diff(0, y[1] - y[0]) ** 2
    # d = FinDiff(1, y[1]-y[0], 2)
    u = X**2 + Y**2

    mat = laplace.matrix(u.shape)

    np.testing.assert_array_almost_equal(
        4 * np.ones_like(X).reshape(-1), mat.dot(u.reshape(-1))
    )

    np.set_printoptions(threshold=thr)
    np.set_printoptions(linewidth=lw)


def test_matrix_2d_mixed():
    x, y = [np.linspace(0, 5, 6), np.linspace(0, 6, 7)]
    X, Y = np.meshgrid(x, y, indexing="ij")
    d2_dxdy = Diff(0, x[1] - x[0]) * Diff(1, y[1] - y[0])
    u = X**2 * Y**2

    mat = d2_dxdy.matrix(u.shape)
    expected = d2_dxdy(u).reshape(-1)

    actual = mat.dot(u.reshape(-1))
    np.testing.assert_array_almost_equal(expected, actual)


def test_matrix_1d_coeffs():
    shape = (11,)
    x = np.linspace(0, 10, 11)
    dx = x[1] - x[0]

    L = x * Diff(0, dx) ** 2

    u = np.random.rand(*shape).reshape(-1)

    actual = L.matrix(shape).dot(u)
    expected = L(u).reshape(-1)
    np.testing.assert_array_almost_equal(expected, actual)
