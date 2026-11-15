from itertools import product

import numpy as np
import pytest

from findiff.stencils import Stencil


@pytest.mark.parametrize("dimensions", [0, 1, 2, 3])
@pytest.mark.parametrize("order", [-1, 0, 1, 3])
def test_taylor_terms_match_product_order(dimensions, order):
    stencil = Stencil.__new__(Stencil)
    stencil.ndims = dimensions
    expected = [
        term
        for term in product(range(order + 1), repeat=dimensions)
        if sum(term) == order
    ]
    assert stencil._multinomial_powers(order) == expected


@pytest.mark.parametrize("dimensions, error", [(-1, ValueError), (1.5, TypeError)])
def test_taylor_terms_reject_invalid_dimensions(dimensions, error):
    stencil = Stencil.__new__(Stencil)
    stencil.ndims = dimensions
    with pytest.raises(error):
        stencil._multinomial_powers(2)


@pytest.mark.parametrize(
    "matrix",
    [
        [[1, 0], [0, 1]],
        [[1, 2], [2, 4]],
        [[1, 0], [0, 1], [1, 1]],
        [[0, 0]],
        [1, 0],
        [[np.inf, 0]],
        [[np.nan, 0]],
        np.empty((1, 0)),
        np.diag([1.0, np.finfo(float).eps]),
        np.diag([1.0, 4 * np.finfo(float).eps]),
    ],
)
def test_row_independence_matches_numpy(matrix):
    stencil = Stencil.__new__(Stencil)
    matrix = np.asarray(matrix, dtype=float)
    try:
        expected = np.linalg.matrix_rank(matrix) == len(matrix)
    except (ValueError, np.linalg.LinAlgError) as error:
        with pytest.raises(type(error)):
            stencil._rows_are_linearly_independent(matrix)
    else:
        assert stencil._rows_are_linearly_independent(matrix) == expected


def test_system_matrix_with_no_offsets_matches_numpy():
    stencil = Stencil.__new__(Stencil)
    stencil.ndims = 1
    stencil.offsets = []
    stencil.max_order = 1
    try:
        np.linalg.matrix_rank(np.empty((1, 0)))
    except ValueError:
        with pytest.raises(ValueError):
            stencil._system_matrix()
    else:
        matrix, terms = stencil._system_matrix()
        assert matrix.size == 0
        assert terms == []


def test_zero_dimensional_stencil():
    stencil = Stencil([()], {(): 2})
    assert stencil.values == {(): 2}
    matrix, terms = stencil._system_matrix()
    np.testing.assert_array_equal(matrix, [[1]])
    assert terms == [()]
