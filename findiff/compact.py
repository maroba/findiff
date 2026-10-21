import warnings

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import splu

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
        # 1D matrices along axis `dim`; the N-D operator is I x ... x L x ... x I,
        # so all work can be done in 1D.
        self._left_matrix = None
        self._right_matrix = None
        self._left_lu = None

    def _prepare(self, shape):
        shape = tuple(shape)
        if self._shape != shape:
            self._shape = shape
            self._calculate_diff_matrix()

    def __call__(self, f):
        f = np.asarray(f)
        self._prepare(f.shape)

        # Solve along axis `dim` for all other grid lines at once.
        f_lines = np.moveaxis(f, self.dim, 0)
        lines_shape = f_lines.shape
        rhs = self._right_matrix.dot(f_lines.reshape(lines_shape[0], -1))
        if np.iscomplexobj(rhs):
            result = self._left_lu.solve(np.ascontiguousarray(rhs.real)) + 1j * (
                self._left_lu.solve(np.ascontiguousarray(rhs.imag))
            )
        else:
            result = self._left_lu.solve(np.ascontiguousarray(rhs, dtype=np.float64))
        return np.moveaxis(result.reshape(lines_shape), 0, self.dim)

    def matrix(self, shape):
        r"""Returns the matrix representation :math:`L^{-1} R` of the compact FD operator.

        Note: For compact schemes the matrix is generally denser than for
        explicit finite differences because inverting the banded L matrix
        introduces fill-in: each row couples all points on its grid line
        along axis `dim`.
        """
        self._prepare(shape)

        M = self._left_lu.solve(self._right_matrix.toarray())
        M = csr_matrix(M)
        if len(self._shape) > 1:
            M = extend_to_ND(M, self.dim, self._shape)
        return csr_matrix(M)

    def _calculate_diff_matrix(self):
        offsets = list(self.scheme.left.keys())
        values = list(self.scheme.left.values())

        n = self._shape[self.dim]
        reach = max(abs(off) for off in list(offsets) + list(self.scheme.right))
        if n < reach:
            raise ValueError(
                f"Grid has {n} points along axis {self.dim}, but the compact scheme "
                f"for derivative order {self.order} reaches {reach} points away "
                f"(left: {sorted(offsets)}, right: {sorted(self.scheme.right)}). "
                f"Use at least {reach} points or a smaller `compact` or `acc`."
            )

        # use lil-type for fast sparse matrix construction:
        L = type(self).creator_function(n, offsets, values, "lil")

        h = self.spacing ** (-self.order)
        coefs = calc_coefs(self.order, self.scheme.right, alphas=self.scheme.left)
        values = [value * h for value in coefs["coefficients"]]
        R = type(self).creator_function(
            self._shape[self.dim], coefs["offsets"], values, mtype="lil"
        )

        L, R = self._modify_boundary_rows(L, R, offsets, coefs, h)

        # convert to sparse matrix type suitable for calculations:
        self._left_matrix = csr_matrix(L)
        self._right_matrix = csr_matrix(R)
        # factorize once, reuse for every call:
        self._left_lu = splu(csc_matrix(L))

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

        # Left boundary: one-sided compact FD (Visbal & Gaitonde, 2002)
        left_rows = list(range(min(left_boundary_size, n)))
        for irow in left_rows:
            self._set_boundary_row(L, R, irow, n, h, coefs["accuracy"], "forward")

        # Right boundary: one-sided compact FD (mirrored)
        for i in range(right_boundary_size):
            irow = n - 1 - i
            if irow in left_rows:
                continue
            self._set_boundary_row(L, R, irow, n, h, coefs["accuracy"], "backward")

        return L, R

    def _set_boundary_row(self, L, R, irow, n, h, interior_acc, side):
        L[irow, :] = 0
        R[irow, :] = 0

        try:
            boundary_alphas, boundary_offsets = self._boundary_scheme(
                irow, n, target_acc=interior_acc
            )
            boundary_coefs = calc_coefs(
                self.order, boundary_offsets, alphas=boundary_alphas
            )
        except np.linalg.LinAlgError:
            # Fallback to explicit one-sided FD
            acc = interior_acc + interior_acc % 2
            warnings.warn(
                f"Could not solve one-sided compact scheme for boundary row {irow}; "
                f"falling back to explicit {side} differences with acc={acc}.",
                RuntimeWarning,
                stacklevel=2,
            )
            L[irow, irow] = 1.0
            explicit = coefficients(self.order, acc)
            for col_off, value in zip(
                explicit[side]["offsets"], explicit[side]["coefficients"]
            ):
                R[irow, irow + col_off] = value * h
            return

        for off, val in boundary_alphas.items():
            L[irow, irow + off] = val
        for off, val in zip(boundary_coefs["offsets"], boundary_coefs["coefficients"]):
            R[irow, irow + off] = val * h

    def _boundary_scheme(self, irow, n, target_acc=None):
        """Compute one-sided compact FD alphas and offsets for a boundary row.

        At boundary rows the interior stencil extends beyond the grid.
        This method keeps only the valid LHS alphas and builds a set of
        RHS offsets that stay within the grid. The RHS stencil is widened
        until it reaches `target_acc` (usually the interior accuracy), as
        far as the grid size permits.
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

        def extend(rhs):
            rmin, rmax = min(rhs), max(rhs)
            if rmax + 1 <= max_valid:
                rhs.append(rmax + 1)
            elif rmin - 1 >= min_valid:
                rhs.append(rmin - 1)
            else:
                return False
            rhs.sort()
            return True

        # Extend to match interior stencil width
        while len(rhs) < len(interior_offsets) and extend(rhs):
            pass

        # One-sided stencils lack the symmetry bonus of the interior scheme,
        # so widen them further until they reach the interior accuracy.
        if target_acc is not None:
            while (
                calc_coefs(self.order, rhs, alphas=alphas)["accuracy"] < target_acc
                and extend(rhs)
            ):
                pass

        return alphas, rhs


class _CompactDiffUniformPeriodic(_CompactDiffUniform):
    creator_function = create_cyclic_band_diagonal
