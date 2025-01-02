from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from findiff.coefs import calc_coefs, coefficients
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

    def __init__(self, left: dict, right: list, periodic=True):
        r"""
        Initializes a CompactScheme instance.

        The compact/implicit finite difference scheme is defined by:

        .. math::
            \sum_k \alpha_k f^{(n)}_{i+k} = \sum_k c_k f_{i+k}

        Args:
            left: dict
                Defines the alphas in the formula above. Keys: k, Values: alpha_k
            right: list|tuple
                Which neighboring points to use (the k's in the formula above)
            periodic: bool
                Whether to use periodic or non-periodic boundary conditions.
        """
        self.left = left
        self.right = right
        self.periodic = periodic


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

        left_boundary_size = max(abs(min(offsets)), abs(min(coefs["offsets"])))
        right_boundary_size = max(max(offsets), max(coefs["offsets"]))
        L[:left_boundary_size, : len(offsets) + left_boundary_size] = 0
        L[-right_boundary_size:, -(len(offsets) + right_boundary_size) :] = 0

        for irow in range(left_boundary_size):
            L[irow, irow] = 1.0

        for irow in range(right_boundary_size):
            L[-irow - 1, -irow - 1] = 1.0

        R[:left_boundary_size, : len(coefs["offsets"]) + left_boundary_size] = 0
        R[-right_boundary_size:, -(len(coefs["offsets"]) + right_boundary_size) :] = 0

        coefs = coefficients(self.order, coefs["accuracy"])

        for irow in range(left_boundary_size):
            for col_off, value in zip(
                coefs["forward"]["offsets"], coefs["forward"]["coefficients"]
            ):
                R[irow, irow + col_off] = value * h

        for irow in range(right_boundary_size):
            for col_off, value in zip(
                coefs["backward"]["offsets"], coefs["backward"]["coefficients"]
            ):
                R[-irow - 1, -irow - 1 + col_off] = value * h
        return L, R


class _CompactDiffUniformPeriodic(_CompactDiffUniform):
    creator_function = create_cyclic_band_diagonal
