import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from findiff.utils import create_cyclic_band_diagonal
from findiff.coefs import calc_coefs


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

        result = spsolve(self._left_matrix, self._right_matrix.dot(f.reshape(-1)))
        return result.reshape(self._shape)

    def _calculate_diff_matrix(self):
        size = np.prod(*self._shape)

        coefs = calc_coefs(self.order, self.scheme.right, alphas=self.scheme.left)
        h = self.spacing ** (-self.order)
        L = create_cyclic_band_diagonal(
            size, list(self.scheme.left.keys()), list(self.scheme.left.values())
        )
        values = [value * h for value in coefs["coefficients"]]
        R = create_cyclic_band_diagonal(size, coefs["offsets"], values)

        self._left_matrix = L
        self._right_matrix = R
