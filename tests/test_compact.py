import numpy as np
import pytest
from sympy import Rational

from findiff import Diff
from findiff.coefs import calc_coefs
from findiff.compact import CompactScheme, _CompactDiffUniformPeriodic
from tests.utils import print_arrays

fmt = {"float_kind": lambda x: f"{x:.3f}"}


def test_compact_differences_coefficients():

    R = Rational
    coefs = calc_coefs(
        1,
        [-3, -2, -1, 0, 1, 2, 3],
        alphas={1: R(1, 3), 0: 1, -1: R(1, 3)},
        symbolic=True,
    )
    c = {off: coef for off, coef in zip(coefs["offsets"], coefs["coefficients"])}

    a = 2 * c[1]
    b = 4 * c[2]

    # The expected values come from Lele, J. Comp. Phys. 103 (1992), p. 17
    assert R(14, 9) == a
    assert R(1, 9) == b
    assert 0 == c[0]

    print(coefs["accuracy"])


def test_compact_differences_diff_set_scheme_implicitly():
    d_dx = Diff(
        0,
        scheme=CompactScheme(
            left={-1: 1 / 3, 0: 1, 1: 1 / 3}, right=[-3, -2, -1, 0, 1, 2, 3]
        ),
    )

    assert d_dx.scheme is not None


def test_compact_differences_diff_set_scheme():
    d_dx = Diff(0)
    scheme = CompactScheme(
        left={-1: 1 / 3, 0: 1, 1: 1 / 3}, right=[-3, -2, -1, 0, 1, 2, 3]
    )

    assert d_dx.scheme is None

    d_dx.set_scheme(scheme)

    assert d_dx.scheme is not None


def test_calculate_diff_matrices():
    scheme = CompactScheme(
        left={-1: 1 / 3, 0: 1, 1: 1 / 3},
        right=[-2, -1, 0, 1, 2],
    )
    x = np.arange(6)
    f = x**2
    np.set_printoptions(linewidth=300)

    differ = _CompactDiffUniformPeriodic(0, 1, 1, scheme)
    differ(f)

    expected = np.array(
        [
            [0.000, 0.778, 0.028, 0.000, -0.028, -0.778],
            [-0.778, 0.000, 0.778, 0.028, 0.000, -0.028],
            [-0.028, -0.778, 0.000, 0.778, 0.028, 0.000],
            [0.000, -0.028, -0.778, 0.000, 0.778, 0.028],
            [0.028, 0.000, -0.028, -0.778, 0.000, 0.778],
            [0.778, 0.028, 0.000, -0.028, -0.778, 0.000],
        ]
    )

    print(
        "\nL=\n",
        np.array2string(
            differ._left_matrix.toarray(),
            separator=", ",
            formatter=fmt,
        ),
    )
    print(
        "\nR=\n",
        np.array2string(
            differ._right_matrix.toarray(),
            separator=", ",
            formatter=fmt,
        ),
    )

    np.testing.assert_array_almost_equal(
        expected, differ._right_matrix.toarray(), decimal=3
    )


def test_compact_differences_diff_apply():
    nx = 40
    x = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    dx = x[1] - x[0]
    d_dx = Diff(0, dx, periodic=True)
    d_dx.set_scheme(
        CompactScheme(
            left={-1: 1 / 3, 0: 1, 1: 1 / 3},
            # left={0: 1},
            right=[-3, -2, -1, 0, 1, 2, 3],
            # right=[-1, 0, 1],
        )
    )
    f = np.exp(np.sin(x))
    expected = np.cos(x) * f
    # f = np.sin(x)
    # expected = np.cos(x)
    actual = d_dx(f)

    print_arrays(
        d_dx._differentiator._right_matrix.toarray(),
        f.reshape(-1, 1),
        actual.reshape(-1, 1),
        expected.reshape(-1, 1),
    )

    # import matplotlib.pyplot as plt
    #
    # plt.semilogy(x, abs((expected - actual)), ".")
    # plt.show()

    np.testing.assert_allclose(expected, actual, atol=1.0e-5)


def test_wrap_around():
    from findiff.utils import create_cyclic_band_diagonal

    print()
    n = 10
    offsets = [-2, -1, 0, 1, 2]
    band_values = [3, 1, 2, 1, 3]

    matrix = create_cyclic_band_diagonal(n, offsets, band_values)
    print(matrix.toarray())


def test_compact_differences_diff_apply_2d():
    num_points = 100
    x = y = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    dx = dy = x[1] - x[0]

    def get_diff(dim):

        d_dx = Diff(dim, dx, periodic=True)
        d_dx.set_scheme(
            CompactScheme(
                left={-1: 1 / 3, 0: 1, 1: 1 / 3},
                right=[-3, -2, -1, 0, 1, 2, 3],
            )
        )
        return d_dx

    D = get_diff(0)
    f = np.exp(np.sin(X))
    expected = np.cos(X) * f
    actual = D(f)

    np.testing.assert_allclose(expected, actual, atol=1.0e-5)

    D = get_diff(1)
    f = np.exp(np.sin(Y))
    expected = np.cos(Y) * f
    actual = D(f)

    np.testing.assert_allclose(expected, actual, atol=1.0e-5)


def test_accuracy():
    coefs = calc_coefs(
        1,
        [-3, -2, -1, 0, 1, 2, 3],
        alphas={1: 1 / 3, 0: 1, -1: 1 / 3},
        symbolic=False,
    )

    assert coefs["accuracy"] == 6
