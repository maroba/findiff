import numpy as np
from numpy.testing import assert_array_almost_equal

from findiff.current import Add, FieldOperator
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

    d_dx = Diff(dx)

    actual = d_dx(f)
    np.testing.assert_array_almost_equal(actual, 3 * x**2, decimal=3)

    actual = d_dx(f, acc=4)
    np.testing.assert_array_almost_equal(actual, 3 * x**2)

    d_dx = Diff(dx, 1, 0)

    actual = d_dx(f, acc=4)
    np.testing.assert_array_almost_equal(actual, 3 * x**2)

    d2_dx2 = Diff(dx, 2, 0)

    actual = d2_dx2(f, acc=4)
    np.testing.assert_array_almost_equal(actual, 6 * x)


def test_chained_derivatives():

    x = np.linspace(0, 1, 100)
    dx = x[1] - x[0]
    f = x**3

    d_dx = Diff(dx)

    d2_dx2 = d_dx * d_dx
    actual = d2_dx2(f, acc=4)
    np.testing.assert_array_almost_equal(actual, 6 * x)


def test_harmonic():

    x = np.linspace(0, 1, 100)
    dx = x[1] - x[0]
    f = x**3

    T = -0.5 * Diff(dx, 2, 0)
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
    fd = Diff(dx, 1, 0)

    ux = fd(u, spacing=dx, acc=4)

    assert_array_almost_equal(ux, ux_ex, decimal=5)

    ny = 80
    y = np.linspace(0, np.pi, ny)
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    u = np.sin(X) * np.sin(Y)
    uxy_ex = np.cos(X) * np.cos(Y)

    fd = Diff(dx, 1, 0) * Diff(dy, 1, 1)

    uxy = fd(u, acc=4)

    assert_array_almost_equal(uxy, uxy_ex, decimal=5)
