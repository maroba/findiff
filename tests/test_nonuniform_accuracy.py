"""Accuracy-order regression tests for non-uniform grids.

On a non-uniform grid the central stencil earns no symmetry bonus, so the
interior scheme for an even-order derivative must carry as many points as the
one-sided schemes to reach the requested order. Before that fix the even-order
interior schemes were one point short and only reached order ``acc - 1`` (the
non-uniform Laplacian was only first-order in the interior). Odd orders were
already correct. These tests pin the order down by polynomial exactness and by
a graded-grid convergence study.
"""

import numpy as np
import pytest

from findiff import Diff


def _graded_grid(n, ratio=2.0, length=3.0):
    """Grid with alternating spacings h, ratio*h, ... (fixed O(1) asymmetry).

    A smoothly mapped grid hides the order loss (its asymmetry vanishes as the
    grid is refined); a graded/boundary-layer mesh does not.
    """
    steps = np.empty(n - 1)
    steps[0::2] = 1.0
    steps[1::2] = ratio
    x = np.concatenate([[0.0], np.cumsum(steps)])
    return x / x[-1] * length


@pytest.mark.parametrize("acc", [2, 4])
@pytest.mark.parametrize("deriv", [1, 2, 3, 4])
def test_nonuniform_polynomial_exactness(deriv, acc):
    """An order-acc scheme is exact for polynomials of degree deriv+acc-1 at
    every grid point. This failed for even derivatives before the fix."""
    x = _graded_grid(40)
    degree = deriv + acc - 1
    p = np.poly1d(np.random.default_rng(0).standard_normal(degree + 1))
    f = p(x)
    op = Diff(0, x, acc=acc) ** deriv if deriv > 1 else Diff(0, x, acc=acc)
    got = op(f)
    exact = p.deriv(deriv)(x)
    scale = max(1.0, np.max(np.abs(exact)))
    assert np.max(np.abs(got - exact)) / scale < 1e-8


@pytest.mark.parametrize("deriv,acc", [(1, 2), (1, 4), (2, 2), (2, 4)])
def test_nonuniform_convergence_order(deriv, acc):
    """Interior convergence order on a graded grid reaches the requested acc.

    Even derivatives dropped to order acc-1 before the central stencil was
    widened; odd derivatives were already correct.
    """

    def f_and_deriv(x):
        f = np.exp(np.sin(2 * x))
        if deriv == 1:
            return f, 2 * np.cos(2 * x) * f
        return f, (2 * np.cos(2 * x)) ** 2 * f - 4 * np.sin(2 * x) * f

    errs = []
    for n in (41, 81, 161):
        x = _graded_grid(n)
        f, exact = f_and_deriv(x)
        op = Diff(0, x, acc=acc) ** deriv if deriv > 1 else Diff(0, x, acc=acc)
        got = op(f)
        b = deriv + acc + 1  # skip a few boundary points
        errs.append(np.max(np.abs(got[b:-b] - exact[b:-b])))
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert orders.min() > acc - 0.4


def test_nonuniform_even_derivative_interior_matches_boundary_order():
    """Interior even-derivative error is no worse than the boundary error.

    The undersized central stencil made the interior *less* accurate than the
    one-sided boundary stencils, which is backwards.
    """
    x = _graded_grid(60)
    f = np.exp(np.sin(2 * x))
    exact = (2 * np.cos(2 * x)) ** 2 * f - 4 * np.sin(2 * x) * f
    got = (Diff(0, x, acc=2) ** 2)(f)
    err = np.abs(got - exact)
    interior = err[5:-5].max()
    boundary = max(err[:5].max(), err[-5:].max())
    assert interior <= boundary
