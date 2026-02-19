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
from sympy import Matrix, Rational


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


def calc_coefs(deriv, offsets, symbolic=False, analytic_inv=False, alphas=None):
    """The working horse for calculating finite difference coefficients on equidistant grids.

    Args:
        deriv: int
            The order of the derivative.
        offsets: list of ints
            Which neighboring points to use.
        symbolic: bool
            Compute symbolically or numerically.
        analytic_inv: bool
            Use analytic inverse for Vandermonde matrix or numerical.
        alphas: dict
            Specifies the implicit scheme to use (the left hand side of the
            system equation in `CompactScheme`). If None, normal (explicit)
            finite differences will be used.

    Returns:
        A dict describing the finite difference scheme.
    """

    if alphas and analytic_inv:
        raise NotImplementedError(
            "compact finite differences for analytic inversion not yet implemented"
        )
    if not alphas:
        alphas = {0: 1}

    if analytic_inv:
        if symbolic:
            solver = SymbolicAnalyticSolver()
        else:
            solver = AnalyticSolver()
    else:
        if symbolic:
            solver = SymbolicSolver()
        else:
            solver = Solver()

    calculator = CoefficientCalculator(deriv, offsets, alphas, solver)

    calculator.solve()
    coefs = list(calculator.sol.values())

    if not symbolic:
        offsets = np.array(offsets)

    return {"coefficients": coefs, "offsets": offsets, "accuracy": calculator.acc}


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
        rhs = _build_rhs(offsets, deriv, alphas={0: 1})

        ret = {
            "coefficients": np.linalg.solve(matrix, rhs),
            "offsets": np.array(offsets),
        }

    elif idx >= len(coords) - num_side:
        matrix = _build_matrix_non_uniform(num_coef - 1, 0, coords, idx)

        offsets = list(range(-num_coef + 1, 1))
        rhs = _build_rhs(offsets, deriv, alphas={0: 1})

        ret = {
            "coefficients": np.linalg.solve(matrix, rhs),
            "offsets": np.array(offsets),
        }

    else:
        matrix = _build_matrix_non_uniform(num_side, num_side, coords, idx)

        offsets = list(range(-num_side, num_side + 1))
        rhs = _build_rhs(offsets, deriv, alphas={0: 1})

        ret = {
            "coefficients": np.linalg.solve(matrix, rhs),
            "offsets": np.array([p for p in range(-num_side, num_side + 1)]),
        }

    return ret


def calc_coefs_non_uni_batched(deriv, acc, coords):
    """Compute FD coefficients for all points on a non-uniform grid using batched linear algebra.

    Instead of solving one linear system per grid point, this groups points by stencil type
    (forward/central/backward) and solves all systems in each group with a single batched
    ``np.linalg.solve`` call.

    :param deriv: int > 0: The derivative order.
    :param acc:  even int > 0: The accuracy order.
    :param coords: 1D numpy.ndarray: the coordinates of the axis.
    :return: dict with 'forward', 'backward', 'center' keys plus 'num_bndry'.
             Each scheme dict contains 'coefficients' (2D array) and 'offsets' (1D array).
    """
    _validate_deriv(deriv)
    _validate_acc(acc)

    num_central = 2 * math.floor((deriv + 1) / 2) - 1 + acc
    num_side = num_central // 2
    num_coef = num_central + 1 if deriv % 2 == 0 else num_central

    N = len(coords)

    fwd_indices = np.arange(min(num_side, N))
    fwd_weights = _solve_non_uni_batched(coords, fwd_indices, 0, num_coef - 1, deriv)

    bwd_start = max(num_side, N - num_side)
    bwd_indices = np.arange(bwd_start, N)
    bwd_weights = _solve_non_uni_batched(coords, bwd_indices, num_coef - 1, 0, deriv)

    ctr_indices = np.arange(num_side, max(num_side, N - num_side))
    ctr_weights = _solve_non_uni_batched(coords, ctr_indices, num_side, num_side, deriv)

    return {
        "forward": {
            "coefficients": fwd_weights,
            "offsets": np.arange(num_coef),
        },
        "backward": {
            "coefficients": bwd_weights,
            "offsets": np.arange(-num_coef + 1, 1),
        },
        "center": {
            "coefficients": ctr_weights,
            "offsets": np.arange(-num_side, num_side + 1),
        },
        "num_bndry": num_side,
    }


def _solve_non_uni_batched(coords, indices, p, q, deriv):
    """Solve for FD coefficients for multiple points with the same stencil structure.

    Builds all Vandermonde-like matrices at once and calls ``np.linalg.solve``
    with a 3-D array of matrices and a 2-D array of right-hand sides.

    :param coords: 1D coordinate array.
    :param indices: 1D integer array of grid-point indices to compute for.
    :param p: number of stencil points to the left.
    :param q: number of stencil points to the right.
    :param deriv: derivative order.
    :return: 2D array of shape (len(indices), stencil_size).
    """
    n = len(indices)
    stencil_size = p + q + 1

    if n == 0:
        return np.empty((0, stencil_size))

    j_offsets = np.arange(-p, q + 1)

    # Coordinate differences: shape (n, stencil_size)
    idx_matrix = indices[:, None] + j_offsets[None, :]
    coord_diffs = coords[idx_matrix] - coords[indices, None]

    # Build Vandermonde-like matrices: shape (n, stencil_size, stencil_size)
    # matrices[i, row, col] = coord_diffs[i, col] ** row
    powers = np.arange(stencil_size)
    matrices = coord_diffs[:, None, :] ** powers[None, :, None]

    # Build RHS: only position `deriv` is non-zero (= deriv!)
    rhs = _build_rhs(list(j_offsets), deriv, alphas={0: 1})
    # np.linalg.solve with 3D A expects b to have shape (..., m, k), not (..., m)
    rhs_col = rhs[:, np.newaxis]  # (stencil_size, 1)
    rhs_batch = np.broadcast_to(rhs_col, (n, stencil_size, 1)).copy()

    return np.linalg.solve(matrices, rhs_batch)[:, :, 0]


def vandermonde_matrix(values):
    A = [([1 for _ in values])]
    for i in range(1, len(values)):
        A.append([j**i for j in values])
    return A


def _build_rhs(
    offsets,
    deriv,
    alphas,
    symbolic=False,
):
    """The right hand side of the equation system matrix"""

    if symbolic:
        frac = lambda num, denom: sympy.Rational(num, denom)  # noqa: E731
    else:
        frac = lambda num, denom: num / denom  # noqa: E731

    b = [0 for _ in offsets]
    for j in range(deriv, len(offsets)):
        b[j] = frac(math.factorial(j), math.factorial(j - deriv)) * sum(
            r ** (j - deriv) * alphas[r] for r in alphas
        )

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


def _validate_acc(acc):
    if acc % 2 == 1 or acc <= 0:
        raise ValueError("Accuracy order acc must be positive EVEN integer")


def _validate_deriv(deriv):
    if deriv < 0:
        raise ValueError("Derive degree must be positive integer")


class InverseVandermondeColumn:

    def __init__(self, symbolic=False):

        self.symbolic = symbolic
        if symbolic:
            self.take = lambda arr, ids: [arr[idx] for idx in ids]  # noqa: E731
            self.prod = sympy.prod
            self.minus = lambda x, arr: [x - val for val in arr]  # noqa: E731
            self.frac = lambda num, denom: sympy.Rational(num, denom)  # noqa: E731
        else:
            self.take = lambda arr, ids: arr[ids]  # noqa: E731
            self.prod = np.prod
            self.minus = lambda x, arr: x - arr  # noqa: E731
            self.frac = lambda num, denom: num / denom  # noqa: E731

    def compute(self, column, offsets):
        """Computes a given column of the inverse of the Vandermonde matrix that
        belongs to given offsets, multiplied by the factorial of the column index.

        This code implements the first equation for b_kj under "Proof 1" in
        https://proofwiki.org/wiki/Inverse_of_Vandermonde_Matrix
        Note that the result has to be transposed because the original Vandermonde
        matrix is defined transposed as compared to _build_matrix.

        Also, we have 0-based indexing here, whereas the formulae in the wiki entry
        are 1-based.
        """

        if not self.symbolic:
            offsets = np.array(offsets)

        n = len(offsets)
        k = column + 1
        inv_vandermonde_column = []
        if k == n:
            # If the number of offsets matches the derivative order + 1, there is a special
            # case, compare the lower part of the bracket in the equation in proofwiki.
            for j in range(n):
                denom = self.prod(self.minus(offsets[j], offsets[:j])) * self.prod(
                    self.minus(offsets[j], offsets[j + 1 :])
                )
                inv_vandermonde_column.append(self.frac(1, denom))
        else:
            # This is the "regular" part of the bracket. First compute the sign that is the
            # same for all entries in the column that we compute
            sign = (-1) ** (n - k)
            for j in range(n):
                # All indices except j
                range_wo_j = list(range(j)) + list(range(j + 1, n))
                # Get all combinations of n-k indices that are ascending and do not contain j
                index_set = combinations(range_wo_j, r=n - k)
                enumerator = sum(
                    self.prod(self.take(offsets, list(m))) for m in index_set
                )
                denominator = self.prod(
                    self.minus(offsets[j], self.take(offsets, range_wo_j))
                )
                inv_vandermonde_column.append(sign * self.frac(enumerator, denominator))

        fact = math.factorial(column)

        result = [val * fact for val in inv_vandermonde_column]
        if self.symbolic:
            return result

        return np.array(result)


class CoefficientCalculator:

    def __init__(self, deriv, offsets, alphas, solver):
        self.deriv = deriv
        self.offsets = offsets
        self.alphas = alphas
        self.solver = solver
        self.sol = None
        self.acc = None

    def solve(self):
        coefs = self.solver.solve(self.deriv, self.offsets, self.alphas)
        self.sol = {off: coef for off, coef in zip(self.offsets, coefs)}
        self.acc = self.solver.acc
        return self.sol


class Solver:

    atol = 1.0e-6

    def __init__(self, wrapper=None):
        self.wrapper = wrapper or NumericMatrixWrapper()
        self.acc = None

    def solve(self, deriv, offsets, alphas):
        sol = np.linalg.solve(
            self.lhs(deriv, offsets), self.rhs(deriv, offsets, alphas)
        )
        self.acc = self.calc_accuracy(deriv, offsets, sol, alphas)
        return sol

    def rhs(self, deriv, offsets, alphas):

        b = [0 for _ in offsets]
        for j in range(deriv, len(offsets)):
            b[j] = self.wrapper.frac(
                math.factorial(j), math.factorial(j - deriv)
            ) * sum(r ** (j - deriv) * alphas[r] for r in alphas)

        return self.wrapper(b)

    def lhs(self, deriv, offsets):
        return self.wrapper(vandermonde_matrix(offsets))

    def calc_accuracy(self, deriv, offsets, coefs, alphas):
        n_plus_s = deriv + 1
        max_n = 999
        break_condition = lambda b: abs(b) > self.atol  # noqa: E731

        while True:
            b = 0
            fac = 1 / math.factorial(n_plus_s)
            for o, coef in zip(offsets, coefs):
                b += coef * o**n_plus_s * fac

            s = n_plus_s - deriv
            fac = 1 / math.factorial(s)
            for o, alpha in alphas.items():
                b -= alpha * fac * o**s

            if break_condition(b):
                break

            n_plus_s += 1
            if n_plus_s > max_n:
                raise Exception("Cannot compute accuracy.")

        return round(n_plus_s - deriv)


class SymbolicSolver(Solver):

    atol = 0

    def __init__(self):
        super().__init__(SymbolicMatrixWrapper())

    def solve(self, deriv, offsets, alphas):
        sol = sympy.linsolve(
            (self.lhs(deriv, offsets), self.rhs(deriv, offsets, alphas))
        )
        sol = list(tuple(sol)[0])
        self.acc = self.calc_accuracy(deriv, offsets, sol, alphas)
        return sol


class AnalyticSolver(Solver):
    symbolic = False

    def solve(self, deriv, offsets, alphas):
        sol = InverseVandermondeColumn(symbolic=self.symbolic).compute(deriv, offsets)
        self.acc = self.calc_accuracy(deriv, offsets, sol, alphas)
        return sol


class SymbolicAnalyticSolver(AnalyticSolver):
    symbolic = True
    atol = 0


class SymbolicMatrixWrapper:
    def __call__(self, arr):
        return Matrix(arr)

    def frac(self, num, denom):
        return Rational(num, denom)


class NumericMatrixWrapper:
    def __call__(self, arr):
        return np.array(arr, dtype=np.float64)

    def frac(self, num, denom):
        return num / denom
