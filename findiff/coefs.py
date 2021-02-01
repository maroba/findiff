"""
This module determines finite difference coefficients for uniform and 
non-uniform grids for any desired even accuracy order.

Most important function:

coefficients(deriv, acc=None, offsets=None, symbolic=False)

to calculate the finite difference coefficients for a given derivative
order and given accuracy order to given offsets.
"""

import math
import numpy as np
import sympy


def coefficients(deriv, acc=None, offsets=None, symbolic=False):
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

    if acc and offsets:
        raise ValueError('acc and offsets cannot both be given')

    if offsets:
        return calc_coefs(deriv, offsets, symbolic)

    _validate_acc(acc)
    ret = {}

    # Determine central coefficients

    num_central = 2 * math.floor((deriv + 1) / 2) - 1 + acc
    num_side = num_central // 2
    offsets = list(range(-num_side, num_side+1))

    ret["center"] = calc_coefs(deriv, offsets, symbolic)

    # Determine forward coefficients

    if deriv % 2 == 0:
        num_coef = num_central + 1
    else:
        num_coef = num_central

    offsets = list(range(num_coef))
    ret["forward"] = calc_coefs(deriv, offsets, symbolic)

    # Determine backward coefficients

    offsets = list(range(-num_coef+1, 1))
    ret["backward"] = calc_coefs(deriv, offsets, symbolic)

    return ret


def calc_coefs(deriv, offsets, symbolic=False):

    matrix = _build_matrix(offsets, symbolic)
    rhs = _build_rhs(offsets, deriv, symbolic)
    if symbolic:
        coefs = sympy.linsolve((matrix, rhs))
        coefs = list(tuple(coefs)[0])
        acc = _calc_accuracy(offsets, coefs, deriv, symbolic)
        return {
            "coefficients": coefs,
            "offsets": offsets,
            "accuracy": acc
        }

    else:
        coefs = np.linalg.solve(matrix, rhs)
        acc = _calc_accuracy(offsets, coefs, deriv, symbolic)

        return {
            "coefficients": coefs,
            "offsets": np.array(offsets),
            "accuracy": acc
        }


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
            "offsets": np.array(offsets)
        }

    elif idx >= len(coords) - num_side:
        matrix = _build_matrix_non_uniform(num_coef - 1, 0, coords, idx)

        offsets = list(range(-num_coef+1, 1))
        rhs = _build_rhs(offsets, deriv)

        ret = {
            "coefficients": np.linalg.solve(matrix, rhs),
            "offsets": np.array(offsets)
        }

    else:
        matrix = _build_matrix_non_uniform(num_side, num_side, coords, idx)

        offsets = list(range(-num_side, num_side+1))
        rhs = _build_rhs(offsets, deriv)

        ret = {
            "coefficients": np.linalg.solve(matrix, rhs),
            "offsets": np.array([p for p in range(-num_side, num_side + 1)])
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
        return np.array(A,dtype='float')


def _build_rhs(offsets, deriv, symbolic=False):
    """The right hand side of the equation system matrix"""

    b = [0 for _ in offsets]
    b[deriv] = math.factorial(deriv)
    if symbolic:
        return sympy.Matrix(b)
    else:
        return np.array(b,dtype='float')


def _build_matrix_non_uniform(p, q, coords, k):
    """Constructs the equation matrix for the finite difference coefficients of non-uniform grids at location k"""
    A = [[1] * (p+q+1)]
    for i in range(1, p + q + 1):
        line = [(coords[k+j] - coords[k])**i for j in range(-p, q+1)]
        A.append(line)
    return np.array(A,dtype='float')


def _calc_accuracy(offsets, coefs, deriv, symbolic=False):

    n = deriv + 1
    max_n = 999
    if symbolic:
        break_cond = lambda b: b != 0
    else:
        break_cond = lambda b: abs(b) > 1.E-6

    while True:
        b = 0
        #for i, coef in enumerate(coefs):
        for o, coef in zip(offsets, coefs):
            #k = min(offsets) + i
            b += coef * o ** n

        if break_cond(b):
            break

        n += 1
        if n > max_n:
            raise Exception('Cannot compute accuracy.')

    return round(n - deriv)


def _validate_acc(acc):
    if acc % 2 == 1 or acc <= 0:
        raise ValueError('Accuracy order acc must be positive EVEN integer')


def _validate_deriv(deriv):
    if deriv < 0:
        raise ValueError('Derive degree must be positive integer')
