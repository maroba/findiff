import numpy as np
import pytest
from sympy import Rational

from findiff import Diff
from findiff.coefs import calc_coefs
from findiff.compact import CompactScheme


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


def test_compact_differences_diff_set_scheme():
    d_dx = Diff(0)
    scheme = CompactScheme(left={-1: 1 / 3, 0: 1, 1: 1 / 3}, right=[-2, -1, 0, 1, 2])

    assert d_dx.scheme is None

    d_dx.set_scheme(scheme)

    assert d_dx.scheme is not None


@pytest.mark.skip()
def test_compact_differences_diff_apply():
    nx = 100
    x = np.linspace(0, 2 * np.pi, nx)
    dx = x[1] - x[0]
    d_dx = Diff(0, dx, periodic=True)
    d_dx.set_scheme(
        CompactScheme(
            left={-1: 1 / 3, 0: 1, 1: 1 / 3},
            right=[-2, -1, 0, 1, 2],
        )
    )
    f = np.exp(np.sin(x))
    expected = np.cos(x) * f
    actual = d_dx(f)

    np.testing.assert_array_almost_equal(expected, actual)
