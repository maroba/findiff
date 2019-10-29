import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve


class PDE(object):

    def __init__(self, lhs, rhs, bcs):
        self.lhs = lhs
        self.rhs = rhs
        self.bcs = bcs
        self._L = None

    def solve(self):

        shape = self.bcs.shape
        if self._L is None:
            self._L = self.lhs.matrix(shape) # expensive operation, so cache it

        L = sparse.lil_matrix(self._L)
        f = self.rhs.reshape(-1, 1)

        nz = list(self.bcs.row_inds())

        L[nz, :] = self.bcs.lhs[nz, :]
        f[nz] = np.array(self.bcs.rhs[nz].toarray()).reshape(-1, 1)

        L = sparse.csr_matrix(L)
        return spsolve(L, f).reshape(shape)


class BoundaryConditions(object):

    def __init__(self, shape):
        self.shape = shape
        siz = np.prod(shape)
        self.long_indices = np.array(list(range(siz))).reshape(shape)
        self.lhs = sparse.lil_matrix((siz, siz))
        self.rhs = sparse.lil_matrix((siz, 1))

    def __setitem__(self, key, value):

        lng_inds = self.long_indices[key]

        if isinstance(value, tuple): # Neumann BC
            op, value = value
            # Avoid calling matrix for the whole grid! Optimize later!
            mat = sparse.lil_matrix(op.matrix(self.shape))
            self.lhs[lng_inds, :] = mat[lng_inds, :]
        else: # Dirichlet BC
            self.lhs[lng_inds, lng_inds] = 1

        if isinstance(value, np.ndarray):
            value = value.reshape(-1)[lng_inds]
            for i, v in zip(lng_inds, value):
                self.rhs[i] = v
        else:
            self.rhs[lng_inds] = value

    def row_inds(self):
        nz_rows, nz_cols = self.lhs.nonzero()
        return nz_rows
