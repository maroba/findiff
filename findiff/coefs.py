import math
import numpy as np


def coefficients(deriv, acc):
    """Calculates the finite difference coefficients for given derivative order and accuracy order"""

    if acc % 2 == 1:
        acc += 1

    ret = {}

    num_central = 2 * math.floor((deriv + 1) / 2) - 1 + acc
    num_side = num_central // 2

    # Determine central coefficients

    matrix = _build_matrix(num_side, num_side, deriv)
    rhs = _build_rhs(num_side, num_side, deriv)

    ret["center"] = {
        "coefficients": np.linalg.solve(matrix, rhs),
        "offsets": np.array([p for p in range(-num_side, num_side+1)])
    }

    # Determine forward coefficients

    if deriv % 2 == 0:
        num_coef = num_central + 1
    else:
        num_coef = num_central

    matrix = _build_matrix(0, num_coef - 1, deriv)
    rhs = _build_rhs(0, num_coef - 1, deriv)

    ret["forward"] = {
        "coefficients": np.linalg.solve(matrix, rhs),
        "offsets": np.array([p for p in range(num_coef)])
    }

    # Determine backward coefficients

    matrix = _build_matrix(num_coef - 1, 0, deriv)
    rhs = _build_rhs(num_coef - 1, 0, deriv)

    ret["backward"] = {
        "coefficients": np.linalg.solve(matrix, rhs),
        "offsets": np.array([p for p in range(-num_coef+1, 1)])
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