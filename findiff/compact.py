import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

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
        L = lil_matrix((size, size))
        R = lil_matrix((size, size))
        L.setdiag(1)
        R.setdiag(1)

        nx = self._shape[self.dim]
        coefs = calc_coefs(self.order, self.scheme.right, alphas=self.scheme.left)
        h = self.spacing ** (-self.order)
        for i in range(nx):
            for off, coef in self.scheme.left.items():
                if 0 < i + off < nx:
                    L[i, i + off] = coef
                elif i + off < 0:
                    L[i, nx + off] = coef
                else:
                    L[i, i + off - nx] = coef

            for off, coef in zip(coefs["offsets"], coefs["coefficients"]):
                if 0 < i + off < nx:
                    R[i, i + off] = coef * h
                elif i + off < 0:
                    R[i, nx + off] = coef * h
                else:
                    R[i, i + off - nx] = coef * h

        self._left_matrix = csr_matrix(L)
        self._right_matrix = csr_matrix(R)
