import math
import warnings

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix
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

    def __init__(self, deriv, left: dict, right: list, coefficients=None):
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
            coefficients: list|tuple|None
                The c_k for the offsets in `right`. If None (default), they are
                determined from the alphas by maximizing the accuracy order.
        """
        self.deriv = deriv
        self.left = left
        self.right = right
        self.coefficients = None if coefficients is None else list(coefficients)
        self._spectral_options = None

    def rhs_coefficients(self, deriv=None):
        """Returns the RHS coefficients c_k (for unit grid spacing) for `deriv`."""
        deriv = self.deriv if deriv is None else deriv
        if self.coefficients is not None and deriv == self.deriv:
            return list(self.coefficients)
        return list(calc_coefs(deriv, self.right, alphas=self.left)["coefficients"])

    def get_accuracy(self, deriv):
        if self.coefficients is not None and deriv == self.deriv:
            return Solver().calc_accuracy(deriv, self.right, self.coefficients, self.left)
        calculator = CoefficientCalculator(deriv, self.right, self.left, Solver())
        calculator.solve()
        return calculator.acc

    def for_deriv(self, deriv):
        """Returns a scheme of the same kind (and LHS width) for another derivative."""
        if deriv == self.deriv:
            return self
        if self._spectral_options is not None:
            return CompactScheme.spectral(deriv, **self._spectral_options)
        return CompactScheme.from_accuracy(
            self.get_accuracy(self.deriv), deriv, len(self.left)
        )

    def modified_wavenumber(self, w):
        r"""Returns the modified wavenumber of the scheme for (scaled) wavenumbers
        `w` in :math:`[0, \pi]`.

        Applying the scheme to :math:`e^{i w x / h}` yields :math:`(i w')^n` times
        the function (n = derivative order); this method returns the real part of
        :math:`(w')^n`, which equals :math:`w^n` for an exact derivative.
        """
        w = np.asarray(w, dtype=np.float64)
        num = sum(
            c * np.exp(1j * off * w)
            for off, c in zip(self.right, self.rhs_coefficients())
        )
        den = sum(a * np.exp(1j * k * w) for k, a in self.left.items())
        return (num / den / 1j**self.deriv).real

    def resolving_efficiency(self, tol=1e-3):
        r"""Returns the fraction of :math:`[0, \pi]` in which the modified
        wavenumber has a relative error below `tol` (Lele, 1992)."""
        w = np.linspace(1e-3, np.pi, 20001)
        exact = w**self.deriv
        bad = np.nonzero(np.abs(self.modified_wavenumber(w) - exact) > tol * exact)[0]
        return float((w[bad[0]] if len(bad) else np.pi) / np.pi)

    @classmethod
    def from_accuracy(cls, acc, deriv, num_left):
        """Creates the compact scheme with the narrowest right-hand side that
        reaches accuracy order `acc` for an LHS bandwidth of `num_left`.

        The LHS alphas and the RHS coefficients are determined jointly to
        maximize the accuracy order for the given stencil widths, as in
        Lele, J. Comp. Phys 103, 16-42 (1992).
        """
        p = num_left // 2
        max_q = 49
        for q in range(1, max_q + 1):
            try:
                scheme = cls._solve_jointly(deriv, p, q)
            except np.linalg.LinAlgError:
                continue
            if scheme.get_accuracy(deriv) >= acc:
                return scheme

        raise Exception(f"Unable to solve compact scheme for given accuracy: {acc}")

    @classmethod
    def _solve_jointly(cls, deriv, p, q):
        """Solves for symmetric LHS alphas (offsets -p..p, alpha_0 = 1) and RHS
        coefficients (offsets -q..q) by matching as many Taylor terms as there
        are unknowns."""
        lhs_offsets = [k for k in range(-p, p + 1) if k != 0]
        rhs_offsets = list(range(-q, q + 1))
        num_unknowns = len(rhs_offsets) + len(lhs_offsets)

        A, b = _taylor_conditions(deriv, rhs_offsets, lhs_offsets, num_unknowns)
        sol = np.linalg.solve(A, b)
        alphas = _symmetric_alphas(lhs_offsets, sol[len(rhs_offsets) :], p)

        return CompactScheme(deriv, alphas, rhs_offsets)

    @classmethod
    def spectral(cls, deriv, num_left, num_right, acc=4, tol=1e-3, max_wavenumber=None):
        r"""Creates a compact scheme with spectral-like resolution (Lele, 1992).

        Instead of spending all free parameters on the accuracy order, only order
        `acc` is imposed. The remaining parameters minimize the error of the
        modified wavenumber in the least-squares sense on
        :math:`[0, w_{max}]`, which resolves short waves much better than the
        scheme of maximal order with the same stencil widths.

        Args:
            deriv: int
                The order of the derivative.
            num_left: int
                Number of LHS points (odd).
            num_right: int
                Number of RHS points (odd).
            acc: int
                Minimum accuracy order.
            tol: float
                Relative error of the modified wavenumber that defines the
                resolving efficiency maximized when `max_wavenumber` is None.
            max_wavenumber: float or None
                Upper end :math:`w_{max} \in (0, \pi)` of the optimization range.
                If None, it is chosen to maximize the resolving efficiency for
                `tol` while keeping the LHS system well-conditioned.
        """
        if num_left % 2 == 0 or num_right % 2 == 0:
            raise ValueError("`num_left` and `num_right` must be odd integers.")
        p, q = num_left // 2, num_right // 2
        options = dict(
            num_left=num_left, num_right=num_right, acc=acc, tol=tol,
            max_wavenumber=max_wavenumber,
        )

        if max_wavenumber is not None:
            scheme = cls._solve_spectral(deriv, p, q, acc, max_wavenumber)
        else:
            scheme, best = None, -1
            for w_max in np.linspace(0.3, 0.98, 35) * np.pi:
                candidate = cls._solve_spectral(deriv, p, q, acc, w_max)
                if _lhs_symbol_min(candidate.left) < _MIN_LHS_SYMBOL:
                    continue
                efficiency = candidate.resolving_efficiency(tol)
                if efficiency > best:
                    scheme, best = candidate, efficiency
            if scheme is None:
                raise ValueError(
                    "No well-conditioned spectral scheme found; "
                    "try other stencil widths or `max_wavenumber`."
                )

        scheme._spectral_options = options
        return scheme

    @classmethod
    def _solve_spectral(cls, deriv, p, q, acc, w_max, num_samples=400):
        lhs_offsets = [k for k in range(-p, p + 1) if k != 0]
        rhs_offsets = list(range(-q, q + 1))
        num_unknowns = len(rhs_offsets) + len(lhs_offsets)
        if acc + deriv >= num_unknowns:
            raise ValueError(
                f"Stencil widths ({2 * p + 1}, {2 * q + 1}) leave no free parameters "
                f"for spectral optimization at acc={acc}; use a wider stencil."
            )

        # equality constraints: Taylor conditions up to order `acc`
        C, d = _taylor_conditions(deriv, rhs_offsets, lhs_offsets, acc + deriv)

        # least squares: sum_j c_j e^{ijw} - (iw)^n sum_k alpha_k e^{ikw} = 0
        w = np.linspace(0, w_max, num_samples)
        target = (1j * w) ** deriv
        rows = np.hstack(
            [
                np.exp(1j * np.outer(w, rhs_offsets)),
                -target[:, None] * np.exp(1j * np.outer(w, lhs_offsets)),
            ]
        )
        A = np.vstack([rows.real, rows.imag])
        b = np.concatenate([target.real, target.imag])

        # solve the equality-constrained least squares problem via KKT system
        K = np.block([[2 * A.T @ A, C.T], [C, np.zeros((len(C), len(C)))]])
        rhs = np.concatenate([2 * A.T @ b, d])
        sol = np.linalg.lstsq(K, rhs, rcond=None)[0][:num_unknowns]
        if not np.allclose(C @ sol, d, atol=1e-8):
            raise ValueError(f"Unable to impose accuracy order {acc}.")

        alphas = _symmetric_alphas(lhs_offsets, sol[len(rhs_offsets) :], p)
        coefs = sol[: len(rhs_offsets)]
        sign = -1 if deriv % 2 else 1
        coefs = (coefs + sign * coefs[::-1]) / 2  # (anti-)symmetrize
        return CompactScheme(deriv, alphas, rhs_offsets, coefficients=coefs)


# Lower bound for min_theta sum_k alpha_k cos(k theta), which keeps the LHS
# matrix of spectral schemes well-conditioned.
_MIN_LHS_SYMBOL = 0.05


def _lhs_symbol_min(alphas):
    theta = np.linspace(0, np.pi, 1001)
    return float(
        min(sum(a * np.cos(k * t) for k, a in alphas.items()) for t in theta)
    )


def _taylor_conditions(deriv, rhs_offsets, lhs_offsets, num_rows):
    """Taylor conditions for the unknowns (c_j for rhs_offsets, alpha_k for
    lhs_offsets, alpha_0 = 1):

        sum_j c_j j^m / m! - sum_{k != 0} alpha_k k^(m-deriv) / (m-deriv)!
            = [m == deriv],   m = 0, ..., num_rows - 1
    """
    num_unknowns = len(rhs_offsets) + len(lhs_offsets)
    A = np.zeros((num_rows, num_unknowns))
    b = np.zeros(num_rows)
    for m in range(num_rows):
        for j, off in enumerate(rhs_offsets):
            A[m, j] = off**m / math.factorial(m)
        if m >= deriv:
            s = m - deriv
            for j, off in enumerate(lhs_offsets):
                A[m, len(rhs_offsets) + j] = -(off**s) / math.factorial(s)
            b[m] = 1.0 if s == 0 else 0.0
    return A, b


def _symmetric_alphas(lhs_offsets, values, p):
    """Builds a symmetric alpha dict (alpha_0 = 1), removing round-off asymmetry."""
    solved = dict(zip(lhs_offsets, values))
    alphas = {0: 1}
    for k in range(1, p + 1):
        alphas[k] = alphas[-k] = (solved[k] + solved[-k]) / 2
    return dict(sorted(alphas.items()))


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
        coefs = {
            "offsets": list(self.scheme.right),
            "coefficients": self.scheme.rhs_coefficients(self.order),
            "accuracy": self.scheme.get_accuracy(self.order),
        }
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


class _CompactDiffNonUniform(_CompactDiffUniform):
    r"""Compact finite differences on a non-uniform grid.

    The left-hand side keeps the alphas of the uniform compact scheme (so the
    banded system stays well-conditioned), while the right-hand side coefficients
    of each row are solved from the actual grid spacings. Without the symmetry of
    a uniform grid, the right-hand side needs ``acc + order`` points (rounded up
    to an odd number) to reach accuracy order ``acc``. Near the boundaries the
    stencil is shifted to stay inside the grid.
    """

    def __init__(self, dim, order, coords, scheme):
        coords = np.asarray(coords, dtype=np.float64)
        steps = np.diff(coords)
        if not (np.all(steps > 0) or np.all(steps < 0)):
            raise ValueError("Coordinates must be strictly monotonic.")

        scheme = scheme.for_deriv(order)
        acc = scheme.get_accuracy(order)

        super().__init__(dim, order, None, scheme)
        self.coords = coords
        self.acc = acc

    def _calculate_diff_matrix(self):
        n = self._shape[self.dim]
        x = self.coords
        if len(x) != n:
            raise ValueError(
                f"Array has {n} points along axis {self.dim}, "
                f"but {len(x)} coordinates were given."
            )

        width = self.acc + self.order
        width += 1 - width % 2
        if n < width:
            raise ValueError(
                f"Grid has {n} points along axis {self.dim}, but the compact scheme "
                f"for derivative order {self.order} on a non-uniform grid needs "
                f"{width} points. Use more points or a smaller `acc`."
            )

        half = width // 2
        L = lil_matrix((n, n))
        R = lil_matrix((n, n))
        for i in range(n):
            alphas = {k: a for k, a in self.scheme.left.items() if 0 <= i + k < n}
            lo = min(max(i - half, 0), n - width)
            cols = range(lo, lo + width)
            coefs = _nonuniform_row_coefs(x, i, cols, alphas, self.order)
            for k, a in alphas.items():
                L[i, i + k] = a
            for j, c in zip(cols, coefs):
                R[i, j] = c

        self._left_matrix = csr_matrix(L)
        self._right_matrix = csr_matrix(R)
        self._left_lu = splu(csc_matrix(L))


def _nonuniform_row_coefs(x, i, cols, alphas, deriv):
    r"""Solves for the RHS coefficients :math:`c_j` of row `i` such that

    .. math::
        \sum_k \alpha_k f^{(d)}(x_{i+k}) = \sum_j c_j f(x_j)

    holds for polynomials of degree ``len(cols) - 1``. Distances are scaled by
    the mean spacing of the stencil for better conditioning.
    """
    cols = list(cols)
    scale = (x[cols[-1]] - x[cols[0]]) / (len(cols) - 1)
    rhs_dist = (x[cols] - x[i]) / scale
    lhs_dist = {k: (x[i + k] - x[i]) / scale for k in alphas}

    num = len(cols)
    A = np.empty((num, num))
    b = np.zeros(num)
    for m in range(num):
        A[m] = rhs_dist**m / math.factorial(m)
        if m >= deriv:
            s = m - deriv
            b[m] = sum(a * lhs_dist[k] ** s for k, a in alphas.items()) / math.factorial(s)

    return np.linalg.solve(A, b) / scale**deriv
