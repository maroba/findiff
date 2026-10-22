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


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("dim", [0, 1, 2])
def test_compact_call_matches_matrix_3d(dim, periodic):
    shape = (12, 9, 7)
    f = np.random.default_rng(0).standard_normal(shape)
    d = Diff(dim, 0.1, compact=3, acc=4, periodic=periodic)

    for op in (d, d**2):
        expected = op.matrix(shape).dot(f.reshape(-1)).reshape(shape)
        np.testing.assert_allclose(op(f), expected, rtol=1e-10, atol=1e-10)


def test_compact_complex_input():
    x = np.linspace(0, 1, 40)
    dx = x[1] - x[0]
    f = np.exp(1j * x)
    d_dx = Diff(0, dx, compact=3, acc=6)

    np.testing.assert_allclose(d_dx(f), 1j * f, atol=1e-6)


def test_compact_matrix_only_couples_along_axis():
    """The N-D matrix must stay block-structured instead of being built densely."""
    shape = (200, 200)
    x = np.linspace(0, 1, shape[0])
    d = Diff(1, x[1] - x[0], compact=3, acc=6)

    M = d.matrix(shape)

    assert M.shape == (40000, 40000)
    assert M.nnz <= shape[0] * shape[1] * shape[1]
    X, Y = np.meshgrid(x, x, indexing="ij")
    np.testing.assert_allclose(
        M.dot(np.sin(Y).reshape(-1)).reshape(shape), np.cos(Y), atol=1e-6
    )


def test_boundary_fallback_warns_and_stays_accurate(monkeypatch):
    import findiff.compact as compact_module

    original = compact_module.calc_coefs

    def failing_for_boundary(deriv, offsets, **kwargs):
        if min(offsets) >= 0 or max(offsets) <= 0:  # one-sided boundary stencil
            raise np.linalg.LinAlgError("singular")
        return original(deriv, offsets, **kwargs)

    monkeypatch.setattr(compact_module, "calc_coefs", failing_for_boundary)

    x = np.linspace(0, 1, 40)
    d_dx = Diff(0, x[1] - x[0], compact=3, acc=4)
    with pytest.warns(RuntimeWarning, match="falling back to explicit"):
        actual = d_dx(np.sin(x))

    np.testing.assert_allclose(actual, np.cos(x), atol=1e-5)


def test_boundary_errors_are_not_swallowed(monkeypatch):
    import findiff.compact as compact_module

    original = compact_module.calc_coefs

    def broken_for_boundary(deriv, offsets, **kwargs):
        if min(offsets) >= 0 or max(offsets) <= 0:  # one-sided boundary stencil
            raise TypeError("unexpected")
        return original(deriv, offsets, **kwargs)

    monkeypatch.setattr(compact_module, "calc_coefs", broken_for_boundary)

    with pytest.raises(TypeError, match="unexpected"):
        Diff(0, 0.1, compact=3, acc=4)(np.ones(20))


@pytest.mark.parametrize("deriv, compact, acc", [(1, 3, 6), (2, 3, 6), (1, 5, 8)])
def test_boundary_rows_reach_interior_accuracy(deriv, compact, acc):
    scheme = CompactScheme.from_accuracy(acc, deriv, compact)
    differ = _CompactDiffUniformNonPeriodic(0, deriv, 1, scheme)
    interior_acc = calc_coefs(deriv, scheme.right, alphas=scheme.left)["accuracy"]

    for irow in range(max(max(scheme.left), max(scheme.right))):
        alphas, offsets = differ._boundary_scheme(irow, 40, target_acc=interior_acc)
        assert calc_coefs(deriv, offsets, alphas=alphas)["accuracy"] >= interior_acc


def test_boundary_convergence_order_matches_interior():
    def boundary_error(n):
        x = np.linspace(0, 2, n)
        f = np.exp(np.sin(x))
        d_dx = Diff(0, x[1] - x[0], compact=3, acc=6)
        return np.abs(d_dx(f) - np.cos(x) * f)[:2].max()

    order = np.log2(boundary_error(41) / boundary_error(81))
    assert order > 5.5


@pytest.mark.parametrize("periodic", [False, True])
def test_grid_too_small_gives_clear_error(periodic):
    d_dx = Diff(0, 0.1, compact=5, acc=10, periodic=periodic)
    reach = max(abs(off) for off in d_dx.scheme.right)

    with pytest.raises(ValueError, match=f"Grid has {reach - 1} points along axis 0"):
        d_dx(np.ones(reach - 1))

    assert np.all(np.isfinite(d_dx(np.ones(reach))))


@pytest.mark.parametrize(
    "deriv, compact, acc, alphas, rhs_width",
    [
        (1, 3, 4, {1: 1 / 4}, 3),  # Pade
        (1, 3, 6, {1: 1 / 3}, 5),  # Lele (1992)
        (1, 5, 8, {1: 4 / 9, 2: 1 / 36}, 5),  # Lele (1992)
        (1, 5, 10, {1: 1 / 2, 2: 1 / 20}, 7),  # Lele (1992)
        (2, 3, 4, {1: 1 / 10}, 3),
        (2, 3, 6, {1: 2 / 11}, 5),  # Lele (1992)
    ],
)
def test_from_accuracy_reproduces_classical_schemes(
    deriv, compact, acc, alphas, rhs_width
):
    scheme = CompactScheme.from_accuracy(acc=acc, deriv=deriv, num_left=compact)

    assert len(scheme.right) == rhs_width
    assert scheme.get_accuracy(deriv) == acc
    for k, alpha in alphas.items():
        assert scheme.left[k] == pytest.approx(alpha, rel=1e-10)
        assert scheme.left[-k] == pytest.approx(alpha, rel=1e-10)


def test_wider_lhs_needs_narrower_rhs():
    widths = [
        len(CompactScheme.from_accuracy(acc=10, deriv=1, num_left=c).right)
        for c in (1, 3, 5)
    ]
    assert widths == sorted(widths, reverse=True)
    assert widths[0] > widths[1] > widths[2]


def test_accuracy_of_high_order_scheme_does_not_overflow():
    scheme = CompactScheme(
        deriv=2, left={-1: 1 / 3, 0: 1, 1: 1 / 3}, right=list(range(-7, 8))
    )
    assert scheme.get_accuracy(2) == 14


def _jittered_grid(n, seed):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)
    x[1:-1] += rng.uniform(-0.3, 0.3, n - 2) * (x[1] - x[0])
    return x


@pytest.mark.parametrize("deriv", [1, 2, 3])
@pytest.mark.parametrize("compact, acc", [(3, 4), (3, 6), (5, 8)])
def test_nonuniform_compact_is_exact_for_polynomials(deriv, compact, acc):
    x = _jittered_grid(30, seed=deriv + acc)
    degree = acc + deriv - 1 + (acc + deriv + 1) % 2  # stencil width - 1
    p = np.polynomial.Polynomial(np.random.default_rng(0).standard_normal(degree + 1))
    expected = p.deriv(deriv)(x)

    actual = (Diff(0, x, compact=compact, acc=acc) ** deriv)(p(x))

    np.testing.assert_allclose(actual, expected, atol=1e-7 * np.abs(expected).max())


@pytest.mark.parametrize("deriv, acc", [(1, 4), (1, 6), (2, 4), (2, 6)])
def test_nonuniform_compact_convergence_on_chebyshev_grid(deriv, acc):
    def error(n):
        x = (1 - np.cos(np.pi * np.linspace(0, 1, n))) / 2
        f = np.sin(3 * x)
        exact = 3 * np.cos(3 * x) if deriv == 1 else -9 * np.sin(3 * x)
        d = Diff(0, x, compact=3, acc=acc) ** deriv
        return np.abs(d(f) - exact)[n // 4 : 3 * n // 4].max()

    assert np.log2(error(21) / error(41)) > acc - 0.5


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_nonuniform_compact_matrix_matches_call_3d(dim):
    xs = [_jittered_grid(n, seed=n) for n in (12, 10, 9)]
    X = np.meshgrid(*xs, indexing="ij")
    f = np.sin(X[0] + 2 * X[1] + X[2])

    for d in (Diff(dim, xs[dim], compact=3, acc=4), Diff(dim, xs[dim], compact=3, acc=4) ** 2):
        expected = d(f)
        actual = d.matrix(f.shape).dot(f.reshape(-1)).reshape(f.shape)
        np.testing.assert_allclose(actual, expected, atol=1e-9 * np.abs(expected).max())


def test_nonuniform_compact_on_uniform_coords_matches_spacing_accuracy():
    x = np.linspace(0, 1, 60)
    f, exact = np.exp(np.sin(3 * x)), 3 * np.cos(3 * x) * np.exp(np.sin(3 * x))

    err_coords = np.abs(Diff(0, x, compact=3, acc=6)(f) - exact).max()
    err_spacing = np.abs(Diff(0, x[1] - x[0], compact=3, acc=6)(f) - exact).max()

    assert err_coords < 2 * err_spacing


def test_nonuniform_compact_decreasing_coords():
    x = np.linspace(1, 0, 40) ** 2
    np.testing.assert_allclose(
        Diff(0, x, compact=3, acc=6)(np.sin(x)), np.cos(x), atol=1e-6
    )


def test_nonuniform_compact_invalid_input():
    with pytest.raises(ValueError, match="strictly monotonic"):
        Diff(0, np.array([0.0, 1.0, 1.0, 2.0]), compact=3)(np.ones(4))

    with pytest.raises(ValueError, match="needs 7 points"):
        Diff(0, np.linspace(0, 1, 5), compact=3, acc=6)(np.ones(5))

    with pytest.raises(ValueError, match="10 coordinates"):
        Diff(0, np.linspace(0, 1, 10), compact=3)(np.ones(12))


@pytest.mark.parametrize("deriv", [1, 2])
@pytest.mark.parametrize("compact, acc", [(3, 4), (5, 4), (5, 6)])
def test_spectral_scheme_resolves_better_than_max_order(deriv, compact, acc):
    spectral = CompactScheme.spectral(deriv, compact, compact + 2, acc=acc)
    max_order = CompactScheme._solve_jointly(deriv, compact // 2, (compact + 2) // 2)

    assert spectral.get_accuracy(deriv) >= acc
    assert spectral.resolving_efficiency() > max_order.resolving_efficiency() + 0.1
    theta = np.linspace(0, np.pi, 201)
    symbol = sum(a * np.cos(k * theta) for k, a in spectral.left.items())
    assert symbol.min() >= 0.05  # LHS stays well-conditioned


def test_modified_wavenumber_matches_application_to_waves():
    scheme = CompactScheme.spectral(1, 5, 7, acc=4)
    n = 64
    x = np.arange(n) * 2 * np.pi / n
    h = x[1] - x[0]
    d_dx = Diff(0, h, scheme=scheme, periodic=True)

    for k in (3, 10, 20):
        w = k * h
        actual = d_dx(np.sin(k * x))
        expected = scheme.modified_wavenumber(w) / h * np.cos(k * x)
        np.testing.assert_allclose(actual, expected, atol=1e-10 * k)


def test_spectral_parameter_and_higher_derivatives():
    n = 64
    x = np.arange(n) * 2 * np.pi / n
    h = x[1] - x[0]
    k = 19  # about 0.6 pi / h
    f = np.sin(k * x)

    for power, exact in ((1, k * np.cos(k * x)), (2, -(k**2) * np.sin(k * x))):
        spectral = Diff(0, h, compact=5, acc=4, spectral=True, periodic=True) ** power
        max_order = Diff(0, h, compact=5, acc=4, periodic=True)
        max_order.set_scheme(CompactScheme._solve_jointly(1, 2, 3))
        max_order = max_order**power

        assert spectral.scheme.deriv == power
        err_spectral = np.abs(spectral(f) - exact).max()
        err_max_order = np.abs(max_order(f) - exact).max()
        assert err_spectral < err_max_order / 5


def test_spectral_with_fixed_max_wavenumber():
    scheme = CompactScheme.spectral(1, 5, 7, acc=4, max_wavenumber=0.6 * np.pi)
    w = np.linspace(0.01, 0.6 * np.pi, 50)
    assert np.abs(scheme.modified_wavenumber(w) - w).max() < 1e-2


def test_explicit_rhs_coefficients_are_used():
    alphas = {-1: 1 / 3, 0: 1, 1: 1 / 3}
    offsets = [-2, -1, 0, 1, 2]
    coefs = calc_coefs(1, offsets, alphas=alphas)["coefficients"]
    perturbed = CompactScheme(1, alphas, offsets, coefficients=[2 * c for c in coefs])

    x = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    d_dx = Diff(0, x[1] - x[0], scheme=perturbed, periodic=True)

    np.testing.assert_allclose(d_dx(np.sin(x)), 2 * np.cos(x), atol=1e-5)


def test_spectral_invalid_input():
    with pytest.raises(ValueError, match="requires `compact`"):
        Diff(0, 0.1, spectral=True)
    with pytest.raises(ValueError, match="no free parameters"):
        CompactScheme.spectral(1, 3, 5, acc=6)
    with pytest.raises(ValueError, match="odd"):
        CompactScheme.spectral(1, 3, 6)
