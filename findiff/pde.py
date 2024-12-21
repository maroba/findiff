"""
This module contains class for solving Partial Differential Equations (PDE)
with Dirichlet and Neumann Boundary Conditions.
"""


import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve


class PDE:
    """
    Representation of a partial differential equation.
    """

    def __init__(self, lhs, rhs, bcs):
        """
        Initializes the PDE.

        You need to specify the left hand side (lhs) in terms of derivatives
        as well as the right hand side in terms of an array.

        Parameters
        ----------
        lhs: FinDiff object or combination of FinDiff objects
            the left hand side of the PDE
        rhs: numpy.ndarray
            the right hand side of the PDE
        bcs: BoundaryConditions
            the boundary conditions for the PDE

        """
        self.lhs = lhs
        self.rhs = rhs
        self.bcs = bcs
        self._L = None

    def solve(self):
        """
        Solves the PDE.

        Returns
        -------
        out: numpy.ndarray
            Array with the solution of the PDE.
        """

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


class BoundaryConditions:
    """
    Represents Dirichlet or Neumann boundary conditions for a PDE.
    """

    def __init__(self, shape):
        """
        Initializes the BoundaryCondition object.

        The BoundaryCondition objects needs information about the
        grid on which to solve the PDE, specifically the shape
        of the (equidistant) grid.

        Parameters
        ----------
        shape: tuple of ints
            the number of grid points in each dimension

        """

        self.shape = shape
        siz = np.prod(shape)
        self.long_indices = np.array(list(range(siz))).reshape(shape)
        self.lhs = sparse.lil_matrix((siz, siz))
        self.rhs = sparse.lil_matrix((siz, 1))

    def __setitem__(self, key, value):
        """
        Sets the boundary condition for specific grid points.

        Parameters
        ----------
        key: int, tuple of ints or slices
            where (on what grid points) to apply the boundary condition.
            Specified by the indices (or slices) of the grid point(s).

        value: Constant or FinDiff object
            the boundary condition to apply. Is a constant (scalar or array)
            for Dirichlet and a FinDiff object for Neumann boundary conditions
        """

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
