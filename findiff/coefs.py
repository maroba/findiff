"""
This module determines finite difference coefficients for uniform and 
non-uniform grids for any desired even accuracy order.

Most important function:

coefficients(deriv, acc=None, offsets=None, symbolic=False)

to calculate the finite difference coefficients for a given derivative
order and given accuracy order to given offsets.
"""

import math
from itertools import combinations

import numpy as np
import sympy


def coefficients(deriv, acc=None, offsets=None, symbolic=False, analytic_inv=False):
    """
    Calculates the finite difference coefficients for given derivative order and accuracy order.

    If acc is given, the coefficients are calculated for central, forward and backward
    schemes resulting in the specified accuracy order.

    If offsets are given, the coefficients are calculated for the offsets as specified
    and the resulting accuracy order is computed.

    *Either* acc *or* offsets must be given.

    Assumes that the underlying grid is uniform. This function is available at the `findiff`
    package level.

    :param deriv: The derivative order.
    :type deriv: int > 0

    :param acc: The accuracy order.
    :type acc:  even int > 0:

    :param offsets: The offsets for which to calculate the coefficients.
    :type offsets: list of ints

    :raises ValueError: if invalid arguments are given

    :return: dict with the finite difference coefficients and corresponding offsets
    """

    _validate_deriv(deriv)

    if acc is not None and offsets:
        raise ValueError("acc and offsets cannot both be given")

    if offsets:
        if deriv >= len(offsets):
            raise ValueError(
                f"can not compute derivative of order {deriv} using {len(offsets)} offsets."
            )
        return calc_coefs(deriv, offsets, symbolic, analytic_inv)

    if acc is None:
        raise ValueError("either acc or offsets has to be given")

    _validate_acc(acc)
    ret = {}
    num_central_coefs = 2 * math.floor((deriv + 1) / 2) - 1 + acc
    num_side_coefs = num_central_coefs // 2

    # Determine central coefficients
    offsets = list(range(-num_side_coefs, num_side_coefs + 1))
    ret["center"] = calc_coefs(deriv, offsets, symbolic, analytic_inv)

    # Determine forward coefficients

    if deriv % 2 == 0:
        num_coef = num_central_coefs + 1
    else:
        num_coef = num_central_coefs

    offsets = list(range(num_coef))
    ret["forward"] = calc_coefs(deriv, offsets, symbolic, analytic_inv)

    # Determine backward coefficients

    offsets = list(range(-num_coef + 1, 1))
    ret["backward"] = calc_coefs(deriv, offsets, symbolic, analytic_inv)

    return ret


def compute_inverse_Vandermonde(column, offsets, symbolic):
    """Computes a given column of the inverse of the Vandermonde matrix that
    belongs to given offsets, multiplied by the factorial of the column index.

    This code implements the first equation for b_kj under "Proof 1" in
    https://proofwiki.org/wiki/Inverse_of_Vandermonde_Matrix
    Note that the result has to be transposed because the original Vandermonde
    matrix is defined transposed as compared to _build_matrix.

    Also, we have 0-based indexing here, whereas the formulae in the wiki entry
    are 1-based.
    """
    if symbolic:
        take = lambda arr, ids: [arr[idx] for idx in ids]  # noqa: E731
        prod = sympy.prod
        minus = lambda x, arr: [x - val for val in arr]  # noqa: E731
    else:
        take = lambda arr, ids: arr[ids]  # noqa: E731
        prod = np.prod
        offsets = np.array(offsets)
        minus = lambda x, arr: x - arr  # noqa: E731

    n = len(offsets)
    k = column + 1
    inv_vandermonde_column = []
    if k == n:
        # If the number of offsets matches the derivative order + 1, there is a special
        # case, compare the lower part of the bracket in the equation in proofwiki.
        for j in range(n):
            denom = prod(minus(offsets[j], offsets[:j])) * prod(
                minus(offsets[j], offsets[j + 1 :])
            )
            inv_vandermonde_column.append(1 / denom)
    else:
        # This is the "regular" part of the bracket. First compute the sign that is the
        # same for all entries in the column that we compute
        sign = (-1) ** (n - k)
        for j in range(n):
            # All indices except j
            range_wo_j = list(range(j)) + list(range(j + 1, n))
            # Get all combinations of n-k indices that are ascending and do not contain j
            index_set = combinations(range_wo_j, r=n - k)
            enumerator = sum(prod(take(offsets, list(m))) for m in index_set)
            denominator = prod(minus(offsets[j], take(offsets, range_wo_j)))
            inv_vandermonde_column.append(sign * enumerator / denominator)

    if symbolic:
        fact = math.factorial(column)
        return [val * fact for val in inv_vandermonde_column]

    return np.array(inv_vandermonde_column) * math.factorial(column)


def calc_coefs(deriv, offsets, symbolic=False, analytic_inv=False):
    if analytic_inv:
        coefs = compute_inverse_Vandermonde(deriv, offsets, symbolic)
    else:
        matrix = _build_matrix(offsets, symbolic)
        rhs = _build_rhs(offsets, deriv, symbolic)
        if symbolic:
            coefs = sympy.linsolve((matrix, rhs))
            coefs = list(tuple(coefs)[0])
        else:
            coefs = np.linalg.solve(matrix, rhs)

    acc = _calc_accuracy(offsets, coefs, deriv, symbolic)

    if not symbolic:
        offsets = np.array(offsets)

    return {"coefficients": coefs, "offsets": offsets, "accuracy": acc}


def coefficients_non_uni(deriv, acc, coords, idx):
    """
    Calculates the finite difference coefficients for given derivative order and accuracy order.
    Assumes that the underlying grid is non-uniform.

    :param deriv: int > 0: The derivative order.

    :param acc:  even int > 0: The accuracy order.

    :param coords:  1D numpy.ndarray: the coordinates of the axis for the partial derivative

    :param idx:  int: index of the grid position where to calculate the coefficients

    :return: dict with the finite difference coefficients and corresponding offsets
    """

    _validate_deriv(deriv)
    _validate_acc(acc)

    num_central = 2 * math.floor((deriv + 1) / 2) - 1 + acc
    num_side = num_central // 2

    if deriv % 2 == 0:
        num_coef = num_central + 1
    else:
        num_coef = num_central

    if idx < num_side:
        matrix = _build_matrix_non_uniform(0, num_coef - 1, coords, idx)

        offsets = list(range(num_coef))
        rhs = _build_rhs(offsets, deriv)

        ret = {
            "coefficients": np.linalg.solve(matrix, rhs),
            "offsets": np.array(offsets),
        }

    elif idx >= len(coords) - num_side:
        matrix = _build_matrix_non_uniform(num_coef - 1, 0, coords, idx)

        offsets = list(range(-num_coef + 1, 1))
        rhs = _build_rhs(offsets, deriv)

        ret = {
            "coefficients": np.linalg.solve(matrix, rhs),
            "offsets": np.array(offsets),
        }

    else:
        matrix = _build_matrix_non_uniform(num_side, num_side, coords, idx)

        offsets = list(range(-num_side, num_side + 1))
        rhs = _build_rhs(offsets, deriv)

        ret = {
            "coefficients": np.linalg.solve(matrix, rhs),
            "offsets": np.array([p for p in range(-num_side, num_side + 1)]),
        }

    return ret


def _build_matrix(offsets, symbolic=False):
    """Constructs the equation system matrix for the finite difference coefficients"""

    A = [([1 for _ in offsets])]
    for i in range(1, len(offsets)):
        A.append([j**i for j in offsets])
    if symbolic:
        return sympy.Matrix(A)
    else:
        return np.array(A, dtype="float")


def _build_rhs(offsets, deriv, symbolic=False):
    """The right hand side of the equation system matrix"""

    b = [0 for _ in offsets]
    b[deriv] = math.factorial(deriv)
    if symbolic:
        return sympy.Matrix(b)
    else:
        return np.array(b, dtype="float")


def _build_matrix_non_uniform(p, q, coords, k):
    """Constructs the equation matrix for the finite difference coefficients of non-uniform grids at location k"""
    A = [[1] * (p + q + 1)]
    for i in range(1, p + q + 1):
        line = [(coords[k + j] - coords[k]) ** i for j in range(-p, q + 1)]
        A.append(line)
    return np.array(A, dtype="float")


def _calc_accuracy(offsets, coefs, deriv, symbolic=False):

    n = deriv + 1
    max_n = 999
    if symbolic:
        break_cond = lambda b: b != 0  # noqa: E731
    else:
        break_cond = lambda b: abs(b) > 1.0e-6  # noqa: E731

    while True:
        b = 0
        # for i, coef in enumerate(coefs):
        for o, coef in zip(offsets, coefs):
            # k = min(offsets) + i
            b += coef * o**n

        if break_cond(b):
            break

        n += 1
        if n > max_n:
            raise Exception("Cannot compute accuracy.")

    return round(n - deriv)


def _validate_acc(acc):
    if acc % 2 == 1 or acc <= 0:
        raise ValueError("Accuracy order acc must be positive EVEN integer")


def _validate_deriv(deriv):
    if deriv < 0:
        raise ValueError("Derive degree must be positive integer")
