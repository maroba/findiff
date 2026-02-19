import numbers
from abc import ABC, abstractmethod
from collections import namedtuple

import numpy as np
from scipy import sparse

from findiff.compact import CompactScheme
from findiff.findiff import build_differentiator
from findiff.grids import GridAxis, make_grid
from findiff.stencils import StencilSet


ErrorEstimate = namedtuple("ErrorEstimate", ["derivative", "error", "extrapolated"])


class Expression(ABC):
    """Represents a differential operator expression."""

    __array_priority__ = 100  # Makes sure custom multiplication is called over numpy's

    def __init__(self, *args, **kwargs):
        self.children = []

    def __repr__(self):
        return str(self)

    @abstractmethod
    def __call__(self, f, *args, **kwargs):
        """Applies the differential operator expression L to array f.

        Parameters
        ----------
        f : numpy.ndarray
            The arrow to differentiate.

        Returns
        -------
        numpy.ndarray
            Array of the same shape as f containing the derivative values.
        """
        pass

    @abstractmethod
    def matrix(self, shape):
        """Returns a matrix representation of the differential operator for a given grid shape."""
        pass

    def stencil(self, shape):
        """Returns a stencil representation of the differential operator for a given grid shape."""
        return StencilSet(self, shape)

    def __add__(self, other):
        """Allows to add differential operator expressions."""
        return Add(self, other)

    def __radd__(self, other):
        """Allows to add differential operator expressions."""
        return Add(self, other)

    def __sub__(self, other):
        """Allows to subtract differential operator expressions."""
        return Add(ScalarOperator(-1) * other, self)

    def __rsub__(self, other):
        """Allows to subtract differential operator expressions."""
        return Add(ScalarOperator(-1) * other, self)

    def __mul__(self, other):
        """Allows to multiply differential operator expressions."""
        return Mul(self, other)

    def __rmul__(self, other):
        """Allows to multiply differential operator expressions."""
        return Mul(other, self)

    @property
    def grid(self):
        """Returns the grid used."""
        return getattr(self, "_grid", None)

    def set_grid(self, grid):
        """Sets the grid for the given differential operator expression.

        Parameters
        ----------
        grid: dict | Grid
            Specifies the grid to use. If a dict is given, an equidistant grid
            is assumed and the dict specifies the spacings along the required axes.
        """
        self._grid = make_grid(grid)
        for child in self.children:
            child.set_grid(self._grid)

    def set_accuracy(self, acc):
        """Sets the requested accuracy for the given differential operator expression.

        Parameters
        ----------
        acc: int
            The accuracy order. Must be a positive, even number.
        """
        self.acc = acc
        for child in self.children:
            child.set_accuracy(acc)

    def estimate_error(self, f, acc=None):
        r"""Estimate truncation error by comparing results at two accuracy orders.

        Computes the derivative at accuracy order *p* and at *p + 2*, then
        uses the pointwise difference as an error estimate for the order-\ *p*
        result.  The higher-order result is also returned as an improved
        ("extrapolated") derivative.

        Parameters
        ----------
        f : numpy.ndarray
            The array to differentiate.
        acc : int or None
            Base accuracy order.  If ``None``, uses the operator's current
            accuracy (default 2).

        Returns
        -------
        ErrorEstimate
            Named tuple with fields:

            - **derivative** – result at the base accuracy order *p*.
            - **error** – estimated pointwise absolute truncation error.
            - **extrapolated** – result at accuracy order *p + 2*.

        Raises
        ------
        NotImplementedError
            If compact finite difference schemes are in use.

        Examples
        --------
        >>> import numpy as np
        >>> from findiff import Diff
        >>> x = np.linspace(0, 2 * np.pi, 200)
        >>> d_dx = Diff(0, x[1] - x[0])
        >>> result = d_dx.estimate_error(np.sin(x))
        >>> result.derivative.shape
        (200,)
        """
        if self._has_compact_scheme():
            raise NotImplementedError(
                "estimate_error does not support compact finite difference schemes"
            )

        original_acc = acc if acc is not None else getattr(self, "acc", 2)

        result_low = self(f, acc=original_acc)
        result_high = self(f, acc=original_acc + 2)

        # Restore operator to its original accuracy
        self.set_accuracy(original_acc)

        error = np.abs(result_low - result_high)
        return ErrorEstimate(result_low, error, result_high)

    def _has_compact_scheme(self):
        """Check whether any node in the expression tree uses a compact scheme."""
        if getattr(self, "scheme", None) is not None:
            return True
        return any(child._has_compact_scheme() for child in self.children)

    def _eliminate_boundary_dofs(self, mat, bc):
        """Extract the interior submatrix by removing boundary DOFs.

        Parameters
        ----------
        mat : scipy.sparse matrix of shape (N, N)
        bc : BoundaryConditions

        Returns
        -------
        interior_mat : scipy.sparse.csr_matrix
        interior_inds : ndarray of int
        """
        N = mat.shape[0]
        boundary_inds = np.unique(list(bc.row_inds()))
        interior_mask = np.ones(N, dtype=bool)
        interior_mask[boundary_inds] = False
        interior_inds = np.where(interior_mask)[0]
        return sparse.csr_matrix(mat[np.ix_(interior_inds, interior_inds)]), interior_inds

    def _reconstruct_eigenvectors(self, vecs, interior_inds, N, shape, k):
        """Pad interior eigenvectors with zeros at boundary DOFs and reshape."""
        full = np.zeros((N, k), dtype=vecs.dtype)
        full[interior_inds, :] = vecs
        return full.reshape(*shape, k)

    def eigs(self, shape, k=6, which='SR', sigma=None, bc=None, M=None, **kwargs):
        r"""Compute k eigenvalues and eigenvectors of this operator.

        Solves :math:`L u = \lambda u` (standard) or
        :math:`L u = \lambda M u` (generalized) using
        ``scipy.sparse.linalg.eigs`` for general (non-symmetric) matrices.

        Parameters
        ----------
        shape : tuple of ints
            Grid shape.
        k : int
            Number of eigenvalues to compute.
        which : str
            Which eigenvalues: 'SR' (smallest real), 'LR', 'SM', 'LM'.
        sigma : float or None
            Shift for shift-invert mode.
        bc : BoundaryConditions or None
            Boundary DOFs are eliminated (homogeneous Dirichlet).
        M : Expression or None
            RHS operator for generalized problem.
        kwargs
            Passed to ``scipy.sparse.linalg.eigs``.

        Returns
        -------
        eigenvalues : ndarray of shape (k,)
            Sorted by real part ascending.
        eigenvectors : ndarray
            Shape ``(\\*shape, k)``.  ``eigenvectors[..., i]`` is the
            *i*-th eigenvector on the grid.
        """
        from scipy.sparse.linalg import eigs as _eigs

        A = self.matrix(shape)
        B = M.matrix(shape) if M is not None else None

        if bc is not None:
            A, interior_inds = self._eliminate_boundary_dofs(A, bc)
            if B is not None:
                B = self._eliminate_boundary_dofs(B, bc)[0]

        eig_kwargs = dict(k=k, which=which, **kwargs)
        if sigma is not None:
            eig_kwargs['sigma'] = sigma
        if B is not None:
            eig_kwargs['M'] = B

        eigenvalues, eigenvectors = _eigs(A, **eig_kwargs)

        idx = np.argsort(eigenvalues.real)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        N = np.prod(shape)
        if bc is not None:
            eigenvectors = self._reconstruct_eigenvectors(
                eigenvectors, interior_inds, N, shape, k
            )
        else:
            eigenvectors = eigenvectors.reshape(*shape, k)

        return eigenvalues, eigenvectors

    def eigsh(self, shape, k=6, which='SM', sigma=None, bc=None, M=None, **kwargs):
        r"""Compute k eigenvalues and eigenvectors of this symmetric operator.

        Like :meth:`eigs`, but uses ``scipy.sparse.linalg.eigsh`` which is
        more efficient for symmetric/Hermitian operators (e.g. Laplacian).
        Eigenvalues are guaranteed real.

        Parameters
        ----------
        shape : tuple of ints
            Grid shape.
        k : int
            Number of eigenvalues to compute.
        which : str
            Which eigenvalues: 'SM' (smallest magnitude), 'LM', 'SA'
            (smallest algebraic), 'LA', 'BE'.
        sigma : float or None
            Shift for shift-invert mode.
        bc : BoundaryConditions or None
            Boundary DOFs are eliminated (homogeneous Dirichlet).
        M : Expression or None
            RHS operator for generalized problem.
        kwargs
            Passed to ``scipy.sparse.linalg.eigsh``.

        Returns
        -------
        eigenvalues : ndarray of shape (k,)
            Real eigenvalues sorted ascending.
        eigenvectors : ndarray
            Shape ``(\\*shape, k)``.  ``eigenvectors[..., i]`` is the
            *i*-th eigenvector on the grid.
        """
        from scipy.sparse.linalg import eigsh as _eigsh

        A = self.matrix(shape)
        B = M.matrix(shape) if M is not None else None

        if bc is not None:
            A, interior_inds = self._eliminate_boundary_dofs(A, bc)
            if B is not None:
                B = self._eliminate_boundary_dofs(B, bc)[0]

        eig_kwargs = dict(k=k, which=which, **kwargs)
        if sigma is not None:
            eig_kwargs['sigma'] = sigma
        if B is not None:
            eig_kwargs['M'] = B

        eigenvalues, eigenvectors = _eigsh(A, **eig_kwargs)

        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        N = np.prod(shape)
        if bc is not None:
            eigenvectors = self._reconstruct_eigenvectors(
                eigenvectors, interior_inds, N, shape, k
            )
        else:
            eigenvectors = eigenvectors.reshape(*shape, k)

        return eigenvalues, eigenvectors


class FieldOperator(Expression):
    """An operator that multiplies an array pointwise."""

    def __init__(self, value):
        super().__init__()
        self.value = value

    def __str__(self):
        if isinstance(self.value, np.ndarray):
            return "f(x)"
        return str(self.value)

    def __call__(self, f, *args, **kwargs):
        if isinstance(f, (numbers.Number, np.ndarray)):
            return self.value * f
        return self.value * super().__call__(f, *args, **kwargs)

    def matrix(self, shape):
        if isinstance(self.value, np.ndarray):
            diag_values = self.value.reshape(-1)
            return sparse.diags(diag_values)
        elif isinstance(self.value, numbers.Number):
            siz = np.prod(shape)
            return sparse.diags(self.value * np.ones(siz))


class ScalarOperator(FieldOperator):
    """A multiple of the identity operator."""

    def __init__(self, value):
        if not isinstance(value, numbers.Number):
            raise ValueError("Expected number, got " + str(type(value)))
        super().__init__(value)

    def __str__(self):
        return str(self.value)

    def matrix(self, shape):
        siz = np.prod(shape)
        mat = sparse.lil_matrix((siz, siz))
        diag = list(range(siz))
        mat[diag, diag] = self.value
        return sparse.csr_matrix(mat)


class Identity(ScalarOperator):
    """The identity operator."""

    def __init__(self):
        super().__init__(1)

    def __str__(self):
        return "I"


class BinaryOperation(Expression):

    @property
    def left(self):
        return self.children[0]

    @property
    def right(self):
        return self.children[1]


class Add(BinaryOperation):
    def __init__(self, left, right):
        if isinstance(left, (numbers.Number, np.ndarray)):
            left = FieldOperator(left)
        if isinstance(right, (numbers.Number, np.ndarray)):
            right = FieldOperator(right)
        super().__init__()
        self.children = [left, right]

    def __call__(self, f, *args, **kwargs):
        return self.left(f, *args, **kwargs) + self.right(f, *args, **kwargs)

    def __str__(self):
        return f"{self.left} + {self.right}"

    def matrix(self, shape):
        return self.left.matrix(shape) + self.right.matrix(shape)


class Mul(BinaryOperation):
    def __init__(self, left, right):
        if isinstance(left, (numbers.Number, np.ndarray)):
            left = FieldOperator(left)
        if isinstance(right, (numbers.Number, np.ndarray)):
            right = FieldOperator(right)
        super().__init__()
        self.children = [left, right]

    def __call__(self, f, *args, **kwargs):
        return self.left(self.right(f, *args, **kwargs), *args, **kwargs)

    def __str__(self):
        left_str = str(self.left)
        right_str = str(self.right)
        if isinstance(self.left, Add):
            left_str = f"({left_str})"
        if isinstance(self.right, Add):
            right_str = f"({right_str})"
        return f"{left_str} * {right_str}"

    def matrix(self, shape):
        return self.left.matrix(shape) * self.right.matrix(shape)


class Diff(Expression):

    DEFAULT_ACC = 2

    def __init__(self, dim, axis: GridAxis = None, acc=DEFAULT_ACC, scheme=None):
        """Initializes a Diff instance.

        Parameters
        ----------
        dim: int
            The 0-based index of the axis along which to take the derivative.
        axis: GridAxis
            The grid axis.
        acc: (optional) int
            The accuracy order to use. Must be a positive even number.
        scheme: CompactScheme or None
            Allows to activate the compact (implicit) finite difference scheme (see documentation
            for CompactScheme). If None, standard explicit finite differences will be used.
        """
        super().__init__()

        self.set_axis(axis)
        self.dim = dim
        self.acc = acc
        self._order = 1
        self._differentiator = None
        self.scheme = scheme

    def set_grid(self, grid):
        super().set_grid(grid)
        self.set_axis(self.grid.get_axis(self.dim))

    def set_axis(self, axis: GridAxis):
        self._axis = axis
        self._differentiator = None

    def __str__(self):
        if self._order == 1:
            return f"d/d{self._axis_label}"
        return f"d{self._order}/d{self._axis_label}{self._order}"

    @property
    def _axis_label(self):
        labels = "xyzwvuts"
        if self.dim < len(labels):
            return labels[self.dim]
        return f"x{self.dim}"

    def set_scheme(self, scheme: CompactScheme = None):
        """Allows to activate using compact (implicit) finite differences."""
        self.scheme = scheme
        self._differentiator = None

    @property
    def axis(self):
        return self._axis

    @property
    def order(self):
        """Returns the order of the derivative."""
        return self._order

    def __call__(self, f, *args, **kwargs):
        """Applies the differential operator."""

        if "acc" in kwargs:
            # allow to pass down new accuracy deep in expression tree
            new_acc = kwargs["acc"]
            if new_acc != self.acc:
                self._differentiator = None
                self.set_accuracy(new_acc)

        if isinstance(f, Expression):
            f = f(*args, **kwargs)

        return self.differentiator(f)

    @property
    def differentiator(self):
        if self._differentiator is None:
            self._differentiator = build_differentiator(
                self.order, self.axis, self.acc, self.scheme
            )
        return self._differentiator

    def matrix(self, shape):
        return self.differentiator.matrix(shape)

    def __pow__(self, power):
        """Returns a Diff instance for a higher order derivative."""
        new_order = self.order * power

        if self.scheme:
            new_scheme = CompactScheme.from_accuracy(
                self.scheme.get_accuracy(self.order), new_order, len(self.scheme.left)
            )
        else:
            new_scheme = None
        new_diff = Diff(self.dim, self.axis, acc=self.acc, scheme=new_scheme)
        new_diff._order = new_order
        return new_diff

    def __mul__(self, other):
        if isinstance(other, Diff) and self.dim == other.dim:
            new_diff = Diff(self.dim, self.axis, acc=self.acc, scheme=self.scheme)
            new_diff._order += other.order
            return new_diff
        return super().__mul__(other)
