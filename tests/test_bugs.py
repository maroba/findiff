import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import findiff
from findiff import FinDiff
from findiff.coefs import coefficients_non_uni


def test_findiff_should_raise_exception_when_applied_to_unevaluated_function():
    def f(x, y):
        return 5 * x**2 - 5 * x + 10 * y**2 - 10 * y  # pragma: no cover

    d_dx = FinDiff(1, 0.01)
    with pytest.raises(ValueError):
        d_dx(f)


def test_matrix_representation_doesnt_work_for_order_greater_2_issue_24():
    x = np.zeros((10))
    d3_dx3 = FinDiff((0, 1, 3))
    mat = d3_dx3.matrix(x.shape)

    assert pytest.approx(mat[0, 0]) == -2.5
    assert pytest.approx(mat[1, 1]) == -2.5
    assert pytest.approx(mat[2, 0]) == -0.5


def test_high_accuracy_results_in_type_error():
    # in issue 25 the following line resulted in a TypeError
    findiff.coefficients(deriv=1, acc=16)


def test_matrix_repr_with_different_accs():
    # issue 28
    shape = (11,)
    d1 = findiff.FinDiff(0, 1, 2).matrix(shape)
    d2 = findiff.FinDiff(0, 1, 2, acc=4).matrix(shape)

    assert np.max(np.abs((d1 - d2).toarray())) > 1

    x = np.linspace(0, 10, 11)
    f = x**2
    df = d2.dot(f)
    np.testing.assert_almost_equal(2 * np.ones_like(f), df)


def test_accuracy_should_be_passed_down_to_stencil():
    # issue 31

    shape = 11, 11
    dx = 1.0
    d1x = FinDiff(0, dx, 1, acc=4)
    stencil1 = d1x.stencil(shape)

    expected = {
        ("L", "L"): {
            (0, 0): -2.083333333333331,
            (1, 0): 3.9999999999999916,
            (2, 0): -2.999999999999989,
            (3, 0): 1.3333333333333268,
            (4, 0): -0.24999999999999858,
        },
        ("L", "C"): {
            (0, 0): -2.083333333333331,
            (1, 0): 3.9999999999999916,
            (2, 0): -2.999999999999989,
            (3, 0): 1.3333333333333268,
            (4, 0): -0.24999999999999858,
        },
        ("L", "H"): {
            (0, 0): -2.083333333333331,
            (1, 0): 3.9999999999999916,
            (2, 0): -2.999999999999989,
            (3, 0): 1.3333333333333268,
            (4, 0): -0.24999999999999858,
        },
        ("C", "L"): {
            (-2, 0): 0.08333333333333333,
            (-1, 0): -0.6666666666666666,
            (1, 0): 0.6666666666666666,
            (2, 0): -0.08333333333333333,
        },
        ("C", "C"): {
            (-2, 0): 0.08333333333333333,
            (-1, 0): -0.6666666666666666,
            (1, 0): 0.6666666666666666,
            (2, 0): -0.08333333333333333,
        },
        ("C", "H"): {
            (-2, 0): 0.08333333333333333,
            (-1, 0): -0.6666666666666666,
            (1, 0): 0.6666666666666666,
            (2, 0): -0.08333333333333333,
        },
        ("H", "L"): {
            (-4, 0): 0.24999999999999958,
            (-3, 0): -1.3333333333333313,
            (-2, 0): 2.9999999999999956,
            (-1, 0): -3.999999999999996,
            (0, 0): 2.0833333333333317,
        },
        ("H", "C"): {
            (-4, 0): 0.24999999999999958,
            (-3, 0): -1.3333333333333313,
            (-2, 0): 2.9999999999999956,
            (-1, 0): -3.999999999999996,
            (0, 0): 2.0833333333333317,
        },
        ("H", "H"): {
            (-4, 0): 0.24999999999999958,
            (-3, 0): -1.3333333333333313,
            (-2, 0): 2.9999999999999956,
            (-1, 0): -3.999999999999996,
            (0, 0): 2.0833333333333317,
        },
    }

    for char_pt in stencil1.data:
        stl = stencil1.data[char_pt]
        assert_dict_almost_equal(expected[char_pt], stl)

    d1x = FinDiff(0, dx, 1, acc=4)
    stencil1 = d1x.stencil(shape)
    for char_pt in stencil1.data:
        stl = stencil1.data[char_pt]
        assert_dict_almost_equal(expected[char_pt], stl)


def test_order_as_numpy_integer():

    order = np.ones(3, dtype=np.int32)[0]
    d_dx = FinDiff(0, 0.1, order)  # raised an AssertionError with the bug

    np.testing.assert_allclose(d_dx(np.linspace(0, 1, 11)), np.ones(11))


def assert_dict_almost_equal(first, second):
    for k in set(first) & set(second):
        assert first[k] == pytest.approx(second[k], rel=1.0e-8)
    # NOTE: missing item(s) should be zero
    for k in set(first) - set(second):
        assert first[k] == pytest.approx(0, abs=1.0e-8)
    for k in set(second) - set(first):
        assert 0 == pytest.approx(second[k], abs=1.0e-8)


class TestIssue90:

    def test_reproduce_issue90(self):

        x = np.array([0.0, 1.0, 1.5, 3.5, 4.0, 6.0])
        f1 = np.array([1, 2, 4, 7, 11, 16])
        f2 = np.sin(x)

        d_dx = FinDiff(0, x, acc=2)
        df_dx1 = d_dx(f1)
        df_dx2 = d_dx(f2)

        grad1 = np.gradient(f1, x, edge_order=2)
        grad2 = np.gradient(f2, x, edge_order=2)

        assert_array_almost_equal(df_dx2, grad2)
        assert_array_almost_equal(df_dx1, grad1)
