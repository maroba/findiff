"""
This module determines finite difference coefficients for uniform and 
non-uniform grids for any desired accuracy order.
"""

import math
import numpy as np


def coefficients(deriv, acc):
    """
    Calculates the finite difference coefficients for given derivative order and accuracy order.
    Assumes that the underlying grid is uniform. This function is available at the `findiff`
    package level.
    
    :param deriv: int > 0: The derivative order.
          
    :param acc:  even int > 0: The accuracy order. 
          
    :return: dict with the finite difference coefficients and corresponding offsets 
    """

    if acc % 2 == 1:
        acc += 1

    ret = {}

    # Determine central coefficients

    num_central = 2 * math.floor((deriv + 1) / 2) - 1 + acc
    num_side = num_central // 2

    ret["center"] = _calc_coef(num_side, num_side, deriv)

    # Determine forward coefficients

    if deriv % 2 == 0:
        num_coef = num_central + 1
    else:
        num_coef = num_central

    ret["forward"] = _calc_coef(0, num_coef - 1, deriv)

    # Determine backward coefficients

    ret["backward"] = _calc_coef(num_coef - 1, 0, deriv)

    return ret


def _calc_coef(left, right, deriv):

    matrix = _build_matrix(left, right, deriv)

    rhs = _build_rhs(left, right, deriv)

    return {
        "coefficients": np.linalg.solve(matrix, rhs),
        "offsets": np.array([p for p in range(-left, right+1)])
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

    if acc % 2 == 1:
        acc += 1

    num_central = 2 * math.floor((deriv + 1) / 2) - 1 + acc
    num_side = num_central // 2

    if deriv % 2 == 0:
        num_coef = num_central + 1
    else:
        num_coef = num_central

    if idx < num_side:
        matrix = _build_matrix_non_uniform(0, num_coef - 1, coords, idx)

        rhs = _build_rhs(0, num_coef - 1, deriv)

        ret = {
            "coefficients": np.linalg.solve(matrix, rhs),
            "offsets": np.array([p for p in range(num_coef)])
        }

    elif idx >= len(coords) - num_side:
        matrix = _build_matrix_non_uniform(num_coef - 1, 0, coords, idx)

        rhs = _build_rhs(num_coef - 1, 0, deriv)

        ret = {
            "coefficients": np.linalg.solve(matrix, rhs),
            "offsets": np.array([p for p in range(-num_coef + 1, 1)])
        }

    else:
        matrix = _build_matrix_non_uniform(num_side, num_side, coords, idx)

        rhs = _build_rhs(num_side, num_side, deriv)

        ret = {
            "coefficients": np.linalg.solve(matrix, rhs),
            "offsets": np.array([p for p in range(-num_side, num_side + 1)])
        }

    return ret


def _build_matrix(p, q, deriv):
    """Constructs the equation system matrix for the finite difference coefficients"""
    A = [([1 for _ in range(-p, q+1)])]
    for i in range(1, p + q + 1):
        A.append([j**i for j in range(-p, q+1)])
    return np.array(A)


def _build_rhs(p, q, deriv):
    """The right hand side of the equation system matrix"""

    b = [0 for _ in range(p+q+1)]
    b[deriv] = math.factorial(deriv)
    return np.array(b)


def _build_matrix_non_uniform(p, q, coords, k):
    """Constructs the equation matrix for the finite difference coefficients of non-uniform grids at location k"""
    A = [[1] * (p+q+1)]
    for i in range(1, p + q + 1):
        line = [(coords[k+j] - coords[k])**i for j in range(-p, q+1)]
        A.append(line)
    return np.array(A)
