import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import splu, spsolve

from findiff.coefs import calc_coefs, coefficients, CoefficientCalculator, Solver
from findiff.utils import (
    create_cyclic_band_diagonal,
    extend_to_ND,
    create_band_diagonal,
)


class CompactScheme:
    r"""
    Represents compact finite difference as in Lele, J. Comp. Phys 103, 16-42 (1992)

    A more appropriate name for the scheme would be "implicite finite differences".

    Normal, i.e. "explicit" finite differences express the n-th derivative as a
    linear combination neighboring function values:

    .. math::
        f^{(n)}_i = \sum_k c_k f_{i+k}

    The compact/implicit finite difference scheme, however uses:
        \sum_k \alpha_k f^{(n)}_{i+k} = \sum_k c_k f_{i+k}

    So, in order to apply implicit finite differences, one has to solve a (sparse) linear
    equation system.

    The big advantage of the implicit scheme is that for the same accuracy order, one needs
    fewer neighboring points than in the explicit schemes.
    """

    def __init__(self, deriv, left: dict, right: list):
        r"""
        Initializes a CompactScheme instance.

        The compact/implicit finite difference scheme is defined by:

        .. math::
            \sum_k \alpha_k f^{(n)}_{i+k} = \sum_k c_k f_{i+k}

        Args:
            deriv: int
                The order of the derivative
            left: dict
                Defines the alphas in the formula above. Keys: k, Values: alpha_k
            right: list|tuple
                Which neighboring points to use (the k's in the formula above)
        """
        self.deriv = deriv
        self.left = left
        self.right = right

    def get_accuracy(self, deriv):
        calculator = CoefficientCalculator(deriv, self.right, self.left, Solver())
        calculator.solve()
        return calculator.acc

    @classmethod
    def from_accuracy(cls, acc, deriv, num_left):
        alphas = {
            off: (1 / num_left) ** abs(off)
            for off in range(-(num_left // 2), (num_left // 2) + 1)
        }
        alphas[0] = 1
        offsets = [-1, 0, 1]
        max_offsets = 99
        while len(offsets) < max_offsets:
            calculator = CoefficientCalculator(deriv, offsets, alphas, Solver())
            try:
                calculator.solve()
                if calculator.acc >= acc:
                    return CompactScheme(deriv, alphas, offsets)
            except np.linalg.LinAlgError:
                pass
            n = len(offsets) // 2 + 1
            offsets = [-n] + offsets + [n]

        raise Exception(f"Unable to solve compact scheme for given accuracy: {acc}")


class _CompactDiffUniform:

    creator_function = None

    def __init__(self, dim, order, spacing, scheme):
        self.dim = dim
        self.order = order
        self.spacing = spacing
        self.scheme = scheme
        self._shape = None
        self._left_matrix = None
        self._right_matrix = None

    def __call__(self, f):
        if self._shape is None or self._shape != f.shape:
            self._shape = f.shape
            self._calculate_diff_matrix()

        result = spsolve(
            csr_matrix(self._left_matrix),
            csr_matrix(self._right_matrix).dot(f.reshape(-1)),
        )
        return result.reshape(self._shape)

    def matrix(self, shape):
        r"""Returns the matrix representation :math:`L^{-1} R` of the compact FD operator.

        Note: For compact schemes the matrix is generally denser than for
        explicit finite differences because inverting the banded L matrix
        introduces fill-in.
        """
        if self._shape is None or self._shape != tuple(shape):
            self._shape = tuple(shape)
            self._calculate_diff_matrix()

        lu = splu(csc_matrix(self._left_matrix))
        M = lu.solve(self._right_matrix.toarray())
        return csr_matrix(M)

    def _calculate_diff_matrix(self):
        offsets = list(self.scheme.left.keys())
        values = list(self.scheme.left.values())

        # use lil-type for fast sparse matrix construction:
        L = type(self).creator_function(self._shape[self.dim], offsets, values, "lil")

        h = self.spacing ** (-self.order)
        coefs = calc_coefs(self.order, self.scheme.right, alphas=self.scheme.left)
        values = [value * h for value in coefs["coefficients"]]
        R = type(self).creator_function(
            self._shape[self.dim], coefs["offsets"], values, mtype="lil"
        )

        L, R = self._modify_boundary_rows(L, R, offsets, coefs, h)

        if len(self._shape) > 1:
            L = extend_to_ND(L, self.dim, self._shape)
            R = extend_to_ND(R, self.dim, self._shape)

        # convert to sparse matrix type suitable for calculations:
        self._left_matrix = csr_matrix(L)
        self._right_matrix = csr_matrix(R)

    def _modify_boundary_rows(self, L, R, offsets, coefs, h):
        # no modification by default:
        return L, R


class _CompactDiffUniformNonPeriodic(_CompactDiffUniform):

    creator_function = create_band_diagonal

    def _modify_boundary_rows(self, L, R, offsets, coefs, h):
        n = L.shape[0]

        interior_lhs_offsets = sorted(self.scheme.left.keys())
        interior_rhs_offsets = sorted(self.scheme.right)

        left_boundary_size = max(abs(min(interior_lhs_offsets)),
                                 abs(min(interior_rhs_offsets)))
        right_boundary_size = max(max(interior_lhs_offsets),
                                  max(interior_rhs_offsets))

        processed = set()

        # Left boundary: one-sided compact FD (Visbal & Gaitonde, 2002)
        for irow in range(left_boundary_size):
            L[irow, :] = 0
            R[irow, :] = 0
            processed.add(irow)

            boundary_alphas, boundary_offsets = self._boundary_scheme(irow, n)
            try:
                boundary_coefs = calc_coefs(
                    self.order, boundary_offsets, alphas=boundary_alphas
                )
                for off, val in boundary_alphas.items():
                    L[irow, irow + off] = val
                for off, val in zip(
                    boundary_coefs["offsets"], boundary_coefs["coefficients"]
                ):
                    R[irow, irow + off] = val * h
            except (np.linalg.LinAlgError, Exception):
                # Fallback to explicit one-sided FD
                L[irow, irow] = 1.0
                explicit = coefficients(self.order, coefs["accuracy"])
                for col_off, value in zip(
                    explicit["forward"]["offsets"],
                    explicit["forward"]["coefficients"],
                ):
                    R[irow, irow + col_off] = value * h

        # Right boundary: one-sided compact FD (mirrored)
        for i in range(right_boundary_size):
            irow = n - 1 - i
            if irow in processed:
                continue
            L[irow, :] = 0
            R[irow, :] = 0

            boundary_alphas, boundary_offsets = self._boundary_scheme(irow, n)
            try:
                boundary_coefs = calc_coefs(
                    self.order, boundary_offsets, alphas=boundary_alphas
                )
                for off, val in boundary_alphas.items():
                    L[irow, irow + off] = val
                for off, val in zip(
                    boundary_coefs["offsets"], boundary_coefs["coefficients"]
                ):
                    R[irow, irow + off] = val * h
            except (np.linalg.LinAlgError, Exception):
                # Fallback to explicit one-sided FD
                L[irow, irow] = 1.0
                explicit = coefficients(self.order, coefs["accuracy"])
                for col_off, value in zip(
                    explicit["backward"]["offsets"],
                    explicit["backward"]["coefficients"],
                ):
                    R[irow, irow + col_off] = value * h

        return L, R

    def _boundary_scheme(self, irow, n):
        """Compute one-sided compact FD alphas and offsets for a boundary row.

        At boundary rows the interior stencil extends beyond the grid.
        This method keeps only the valid LHS alphas and builds a set of
        RHS offsets that stay within the grid, matching the width of the
        interior stencil so that accuracy is preserved.
        """
        interior_alphas = self.scheme.left
        interior_offsets = self.scheme.right

        # Valid offset range from this row
        min_valid = -irow
        max_valid = n - 1 - irow

        # Keep interior alphas that fall inside the grid
        alphas = {}
        for off, val in interior_alphas.items():
            if min_valid <= off <= max_valid:
                alphas[off] = val
        if 0 not in alphas:
            alphas[0] = 1

        # Build RHS offsets within valid range
        rhs = [off for off in interior_offsets if min_valid <= off <= max_valid]

        # Extend to match interior stencil width
        target = len(interior_offsets)
        while len(rhs) < target:
            rmin, rmax = min(rhs), max(rhs)
            if rmax + 1 <= max_valid:
                rhs.append(rmax + 1)
            elif rmin - 1 >= min_valid:
                rhs.append(rmin - 1)
            else:
                break
            rhs.sort()

        return alphas, rhs


class _CompactDiffUniformPeriodic(_CompactDiffUniform):
    creator_function = create_cyclic_band_diagonal
