"""
This module contains class for solving Partial Differential Equations (PDE)
with Dirichlet, Neumann and Robin Boundary Conditions.
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
    Represents boundary conditions for a PDE.

    Supports Dirichlet, Neumann and Robin (mixed) boundary conditions.

    Dirichlet:
        ``bc[index] = value``

    Neumann:
        ``bc[index] = (diff_operator, value)``

    Robin (:math:`\\alpha u + \\beta \\partial u / \\partial n = g`):
        ``bc[index] = (alpha, diff_operator, beta, value)``

    Alternatively, Robin BCs can be specified using the operator-tuple
    form ``bc[index] = (alpha * Identity() + beta * diff_operator, value)``.
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

        value:
            The boundary condition to apply.

            - **Dirichlet**: a scalar or array — ``bc[idx] = g``
            - **Neumann**: a 2-tuple ``(diff_op, g)`` — ``bc[idx] = (Diff(0, dx), g)``
            - **Robin**: a 4-tuple ``(alpha, diff_op, beta, g)`` representing
              :math:`\\alpha u + \\beta \\frac{\\partial u}{\\partial n} = g`.
        """
        from findiff.operators import Identity

        lng_inds = self.long_indices[key]

        if isinstance(value, tuple) and len(value) == 4:
            # Robin BC: (alpha, diff_op, beta, rhs_value)
            alpha, diff_op, beta, rhs_value = value
            robin_op = alpha * Identity() + beta * diff_op
            mat = sparse.lil_matrix(robin_op.matrix(self.shape))
            self.lhs[lng_inds, :] = mat[lng_inds, :]
            value = rhs_value
        elif isinstance(value, tuple):
            # Neumann BC: (diff_op, rhs_value)
            op, value = value
            mat = sparse.lil_matrix(op.matrix(self.shape))
            self.lhs[lng_inds, :] = mat[lng_inds, :]
        else:
            # Dirichlet BC: scalar or array
            self.lhs[lng_inds, :] = 0
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
