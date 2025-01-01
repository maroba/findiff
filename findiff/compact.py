from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from findiff.coefs import calc_coefs
from findiff.utils import create_cyclic_band_diagonal, extend_to_ND


class CompactScheme:

    def __init__(self, left: dict, right: list):
        self.left = left
        self.right = right


class _CompactDiffUniformPeriodic:

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

        L = create_cyclic_band_diagonal(
            self._shape[self.dim],
            list(self.scheme.left.keys()),
            list(self.scheme.left.values()),
        )

        h = self.spacing ** (-self.order)
        coefs = calc_coefs(self.order, self.scheme.right, alphas=self.scheme.left)
        values = [value * h for value in coefs["coefficients"]]
        R = create_cyclic_band_diagonal(self._shape[self.dim], coefs["offsets"], values)

        if len(self._shape) > 1:
            L = extend_to_ND(L, self.dim, self._shape)
            R = extend_to_ND(R, self.dim, self._shape)

        self._left_matrix = L
        self._right_matrix = R
