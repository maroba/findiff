import numpy as np
import pytest
from sympy import Rational

from findiff import Diff
from findiff.coefs import calc_coefs
from findiff.compact import (
    CompactScheme,
    _CompactDiffUniformPeriodic,
    _CompactDiffUniformNonPeriodic,
)
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


def test_compact_differences_coefficients_second_deriv():
    alpha = 2 / 11
    coefs = calc_coefs(
        2,
        [-2, -1, 0, 1, 2],
        alphas={1: alpha, 0: 1, -1: alpha},
        symbolic=False,
    )
    c = {off: coef for off, coef in zip(coefs["offsets"], coefs["coefficients"])}

    a = c[1]
    b = 4 * c[2]

    # The expected values come from Lele, J. Comp. Phys. 103 (1992), p. 19
    assert 12 / 11 == pytest.approx(a)
    assert 3 / 11 == pytest.approx(b)

    assert 6 == coefs["accuracy"]
    print(coefs["accuracy"])


def test_compact_differences_diff_set_scheme_implicitly():
    d_dx = Diff(
        0,
        scheme=CompactScheme(
            deriv=1, left={-1: 1 / 3, 0: 1, 1: 1 / 3}, right=[-3, -2, -1, 0, 1, 2, 3]
        ),
    )

    assert d_dx.scheme is not None


def test_compact_differences_diff_set_scheme():
    d_dx = Diff(0)
    scheme = CompactScheme(
        deriv=1, left={-1: 1 / 3, 0: 1, 1: 1 / 3}, right=[-3, -2, -1, 0, 1, 2, 3]
    )

    assert d_dx.scheme is None

    d_dx.set_scheme(scheme)

    assert d_dx.scheme is not None


def test_calculate_diff_matrices():
    scheme = CompactScheme(
        deriv=1,
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
                deriv=1,
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


def test_compact_differences_diff_apply_2d_nonperiodic():
    num_points = 100
    x = y = np.linspace(0, 2 * np.pi, num_points)
    X, Y = np.meshgrid(x, y, indexing="ij")

    dx = dy = x[1] - x[0]

    def get_diff(dim):

        d_dx = Diff(dim, dx, periodic=False)
        d_dx.set_scheme(
            CompactScheme(
                deriv=1,
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
        [-2, -1, 0, 1, 2],
        alphas={1: 1 / 3, 0: 1, -1: 1 / 3},
        symbolic=False,
    )

    assert coefs["accuracy"] == 6  # according to Lele 1992

    # regular finite differences have lower accuracy... that's
    # the benefit from implicitness...:

    coefs = calc_coefs(
        1,
        [-2, -1, 0, 1, 2],
        alphas={0: 1},
        symbolic=False,
    )

    assert coefs["accuracy"] == 4


def test_nonperiodic_uniform():
    scheme = CompactScheme(
        deriv=1,
        left={1: 1 / 3, 0: 1, -1: 1 / 3},
        right=[-2, -1, 0, 1, 2],
    )
    x = np.linspace(0, 1, 60)
    dx = x[1] - x[0]
    f = np.sin(x)
    d_dx = Diff(0, dx, scheme=scheme)
    d2_dx2 = d_dx**2

    np.testing.assert_array_almost_equal(d_dx(f), np.cos(x))
    np.testing.assert_array_almost_equal(d2_dx2(f), -np.sin(x))


def test_nonperiodic_matrices():
    scheme = CompactScheme(
        deriv=1,
        left={1: 1 / 3, 0: 1, -1: 1 / 3},
        right=[-2, -1, 0, 1, 2],
    )
    differ = _CompactDiffUniformNonPeriodic(dim=0, order=1, spacing=1, scheme=scheme)
    f = np.ones(10, dtype=np.float64)
    differ(f)
    L = differ._left_matrix.toarray()

    # With one-sided compact FD at boundaries, L retains the compact
    # tridiagonal structure at all rows (not identity at boundaries):
    # - Row 0: [1, 1/3, 0, ...] (truncated LHS, offset -1 not available)
    # - Row 1: [1/3, 1, 1/3, 0, ...] (full interior scheme, centered)
    # - Interior rows: standard tridiagonal
    # - Row 8: [..., 0, 1/3, 1, 1/3] (full interior scheme, centered)
    # - Row 9: [..., 0, 1/3, 1] (truncated LHS, offset +1 not available)

    # Verify diagonal is 1 everywhere
    np.testing.assert_array_almost_equal(np.diag(L), np.ones(10))

    # Boundary rows should NOT be identity (compact structure preserved)
    assert L[0, 1] == pytest.approx(1 / 3, abs=1e-6)
    assert L[9, 8] == pytest.approx(1 / 3, abs=1e-6)

    # Interior rows have standard tridiagonal structure
    for i in range(2, 8):
        assert L[i, i - 1] == pytest.approx(1 / 3, abs=1e-6)
        assert L[i, i] == pytest.approx(1.0, abs=1e-6)
        assert L[i, i + 1] == pytest.approx(1 / 3, abs=1e-6)


def test_scheme_from_accuracy():
    # First derivative, tridiagonal compact (Lele 1992, alpha=1/3)
    scheme = CompactScheme.from_accuracy(acc=6, deriv=1, num_left=3)

    # Check the generated alphas
    assert scheme.left[-1] == pytest.approx(1 / 3)
    assert scheme.left[0] == 1
    assert scheme.left[1] == pytest.approx(1 / 3)

    # Check accuracy meets the request
    assert scheme.get_accuracy(1) >= 6

    # Verify it gives correct numerical results
    x = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    dx = x[1] - x[0]
    d_dx = Diff(0, dx, scheme=scheme, periodic=True)
    f = np.sin(x)
    np.testing.assert_allclose(d_dx(f), np.cos(x), atol=1e-5)

    # Second derivative
    scheme2 = CompactScheme.from_accuracy(acc=4, deriv=2, num_left=3)
    assert scheme2.get_accuracy(2) >= 4


def test_compact_differences_diff_apply():
    nx = 40
    x = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    dx = x[1] - x[0]
    d_dx = Diff(0, dx, periodic=True)
    d_dx.set_scheme(
        CompactScheme(
            deriv=1,
            left={-1: 1 / 3, 0: 1, 1: 1 / 3},
            right=[-3, -2, -1, 0, 1, 2, 3],
        )
    )
    f = np.exp(np.sin(x))
    expected = np.cos(x) * f
    actual = d_dx(f)

    print_arrays(
        d_dx._differentiator._right_matrix.toarray(),
        f.reshape(-1, 1),
        actual.reshape(-1, 1),
        expected.reshape(-1, 1),
    )

    np.testing.assert_allclose(expected, actual, atol=1.0e-5)


def test_periodic_uniform_shortcut():
    scheme = CompactScheme(
        deriv=1,
        left={1: 1 / 3, 0: 1, -1: 1 / 3},
        right=[-3, -2, -1, 0, 1, 2, 3],
    )

    x = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    dx = x[1] - x[0]
    f = np.exp(np.sin(x))
    d_dx = Diff(0, dx, scheme=scheme, periodic=True)
    d_dx_shortcut = Diff(0, dx, compact=3, acc=6, periodic=True)

    expected = d_dx(f)
    actual = d_dx_shortcut(f)

    np.testing.assert_allclose(expected, actual, atol=1.0e-12)


def test_compact_scheme_exported():
    from findiff import CompactScheme as CS

    scheme = CS(deriv=1, left={-1: 1 / 3, 0: 1, 1: 1 / 3}, right=[-2, -1, 0, 1, 2])
    assert scheme.deriv == 1


def test_matrix_periodic():
    """matrix() on a periodic compact scheme should give the same result as __call__."""
    scheme = CompactScheme(
        deriv=1,
        left={-1: 1 / 3, 0: 1, 1: 1 / 3},
        right=[-2, -1, 0, 1, 2],
    )
    x = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    dx = x[1] - x[0]
    d_dx = Diff(0, dx, scheme=scheme, periodic=True)
    f = np.sin(x)

    result_call = d_dx(f)
    M = d_dx.matrix((30,))
    result_matrix = M.dot(f)

    np.testing.assert_allclose(result_call, result_matrix, atol=1e-12)


def test_matrix_nonperiodic():
    """matrix() on a non-periodic compact scheme should give same result as __call__."""
    scheme = CompactScheme(
        deriv=1,
        left={-1: 1 / 3, 0: 1, 1: 1 / 3},
        right=[-2, -1, 0, 1, 2],
    )
    x = np.linspace(0, 1, 40)
    dx = x[1] - x[0]
    d_dx = Diff(0, dx, scheme=scheme, periodic=False)
    f = np.sin(x)

    result_call = d_dx(f)
    M = d_dx.matrix((40,))
    result_matrix = M.dot(f)

    np.testing.assert_allclose(result_call, result_matrix, atol=1e-12)


def test_one_sided_compact_boundary_accuracy():
    """One-sided compact FD at boundaries should be more accurate than explicit FD."""
    scheme = CompactScheme(
        deriv=1,
        left={-1: 1 / 3, 0: 1, 1: 1 / 3},
        right=[-2, -1, 0, 1, 2],
    )

    for n in [30, 60, 120]:
        x = np.linspace(0, 1, n)
        dx = x[1] - x[0]
        f = np.sin(x)
        expected = np.cos(x)

        d_dx = Diff(0, dx, scheme=scheme, periodic=False)
        actual = d_dx(f)

        # The derivative should be accurate to at least 1e-4 everywhere
        np.testing.assert_allclose(actual, expected, atol=1e-4)


def test_one_sided_compact_boundary_structure():
    """Boundary rows of L should retain compact structure, not be identity."""
    scheme = CompactScheme(
        deriv=1,
        left={-1: 1 / 3, 0: 1, 1: 1 / 3},
        right=[-2, -1, 0, 1, 2],
    )
    differ = _CompactDiffUniformNonPeriodic(dim=0, order=1, spacing=1, scheme=scheme)
    f = np.ones(20, dtype=np.float64)
    differ(f)
    L = differ._left_matrix.toarray()

    # Row 0: should have compact coupling with row 1
    assert L[0, 0] == pytest.approx(1.0)
    assert L[0, 1] == pytest.approx(1 / 3, abs=1e-6)

    # Last row: should have compact coupling with second-to-last row
    assert L[-1, -1] == pytest.approx(1.0)
    assert L[-1, -2] == pytest.approx(1 / 3, abs=1e-6)


def test_one_sided_compact_second_derivative():
    """One-sided compact FD should work for second derivatives via powering."""
    scheme = CompactScheme(
        deriv=1,
        left={-1: 1 / 3, 0: 1, 1: 1 / 3},
        right=[-2, -1, 0, 1, 2],
    )
    x = np.linspace(0, 1, 80)
    dx = x[1] - x[0]
    f = np.sin(x)
    expected = -np.sin(x)

    d_dx = Diff(0, dx, scheme=scheme, periodic=False)
    d2_dx2 = d_dx ** 2
    actual = d2_dx2(f)

    np.testing.assert_allclose(actual, expected, atol=1e-4)
