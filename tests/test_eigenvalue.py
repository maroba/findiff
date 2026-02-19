"""Tests for eigenvalue problem support (eigs/eigsh on Expression).

Covers:
- Return shapes (1D, 2D)
- 1D Laplacian with Dirichlet BCs (exact eigenvalues known)
- Quantum harmonic oscillator
- 2D Laplacian with Dirichlet BCs
- Periodic operator without BC elimination
- eigs vs eigsh consistency for symmetric operators
- Generalized eigenvalue problem L*u = lambda*M*u
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from findiff import Diff, Identity, BoundaryConditions


class TestEigsBasic:

    def test_methods_exist(self):
        L = Diff(0, 0.1) ** 2
        assert hasattr(L, 'eigs')
        assert hasattr(L, 'eigsh')

    def test_return_shape_1d(self):
        n = 50
        shape = (n,)
        dx = 1.0 / (n - 1)
        L = Diff(0, dx) ** 2
        k = 4

        bc = BoundaryConditions(shape)
        bc[0] = 0
        bc[-1] = 0

        vals, vecs = L.eigsh(shape, k=k, which='SM', bc=bc)
        assert vals.shape == (k,)
        assert vecs.shape == (n, k)

    def test_return_shape_2d(self):
        nx, ny = 15, 15
        shape = (nx, ny)
        dx = dy = 1.0 / (nx - 1)
        L = Diff(0, dx) ** 2 + Diff(1, dy) ** 2
        k = 3

        bc = BoundaryConditions(shape)
        bc[0, :] = 0
        bc[-1, :] = 0
        bc[:, 0] = 0
        bc[:, -1] = 0

        vals, vecs = L.eigsh(shape, k=k, which='SM', bc=bc)
        assert vals.shape == (k,)
        assert vecs.shape == (nx, ny, k)

    def test_return_shape_no_bc(self):
        n = 32
        shape = (n,)
        dx = 2 * np.pi / n
        L = Diff(0, dx, periodic=True) ** 2
        k = 4

        vals, vecs = L.eigsh(shape, k=k, which='SM')
        assert vals.shape == (k,)
        assert vecs.shape == (n, k)


class TestEigsh1DDirichlet:
    """u'' = lambda*u on [0, pi], u(0) = u(pi) = 0.

    Exact eigenvalues: lambda_n = -n^2 for n = 1, 2, 3, ...
    Exact eigenvectors: sin(n*x)
    """

    def test_eigenvalues(self):
        n = 201
        shape = (n,)
        x = np.linspace(0, np.pi, n)
        dx = x[1] - x[0]
        L = Diff(0, dx) ** 2

        bc = BoundaryConditions(shape)
        bc[0] = 0
        bc[-1] = 0

        vals, vecs = L.eigsh(shape, k=5, which='SM', bc=bc)
        expected = np.array([-25, -16, -9, -4, -1], dtype=float)
        assert_array_almost_equal(np.sort(vals), expected, decimal=1)

    def test_eigenvectors_boundary_zeros(self):
        n = 101
        shape = (n,)
        x = np.linspace(0, np.pi, n)
        dx = x[1] - x[0]
        L = Diff(0, dx) ** 2

        bc = BoundaryConditions(shape)
        bc[0] = 0
        bc[-1] = 0

        vals, vecs = L.eigsh(shape, k=3, which='SM', bc=bc)

        # Eigenvectors must be zero at boundary points
        assert_array_almost_equal(vecs[0, :], 0)
        assert_array_almost_equal(vecs[-1, :], 0)

    def test_first_eigenvector_shape(self):
        """First eigenvector should resemble sin(x)."""
        n = 201
        shape = (n,)
        x = np.linspace(0, np.pi, n)
        dx = x[1] - x[0]
        L = Diff(0, dx) ** 2

        bc = BoundaryConditions(shape)
        bc[0] = 0
        bc[-1] = 0

        vals, vecs = L.eigsh(shape, k=1, which='SM', bc=bc)

        # The eigenvector for lambda=-1 is sin(x), up to sign and normalization
        v = vecs[:, 0]
        expected = np.sin(x)
        # Normalize both to compare shape
        v = v / np.max(np.abs(v)) * np.sign(v[len(v) // 2])
        expected = expected / np.max(np.abs(expected))
        assert_array_almost_equal(v, expected, decimal=2)


class TestEigshHarmonicOscillator:
    """Quantum harmonic oscillator: -0.5*u'' + 0.5*x^2*u = E*u.

    Exact eigenvalues: E_n = n + 0.5 for n = 0, 1, 2, ...
    """

    def test_eigenvalues(self):
        n = 300
        shape = (n,)
        x = np.linspace(-8, 8, n)
        dx = x[1] - x[0]

        T = -0.5 * Diff(0, dx) ** 2
        V = 0.5 * x**2 * Identity()
        H = T + V

        bc = BoundaryConditions(shape)
        bc[0] = 0
        bc[-1] = 0

        vals, vecs = H.eigsh(shape, k=6, which='SA', bc=bc)
        expected = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        assert_array_almost_equal(vals, expected, decimal=1)


class TestEigsh2DDirichlet:
    """2D Laplacian on [0, pi]^2 with Dirichlet BCs.

    Exact eigenvalues: lambda_{m,n} = -(m^2 + n^2) for m, n = 1, 2, ...
    Lowest: -2, -5, -5, -8
    """

    def test_eigenvalues(self):
        nx = ny = 41
        shape = (nx, ny)
        x = np.linspace(0, np.pi, nx)
        y = np.linspace(0, np.pi, ny)
        dx, dy = x[1] - x[0], y[1] - y[0]

        L = Diff(0, dx) ** 2 + Diff(1, dy) ** 2

        bc = BoundaryConditions(shape)
        bc[0, :] = 0
        bc[-1, :] = 0
        bc[:, 0] = 0
        bc[:, -1] = 0

        vals, vecs = L.eigsh(shape, k=4, which='SM', bc=bc)
        expected = np.array([-8, -5, -5, -2], dtype=float)
        np.testing.assert_allclose(np.sort(vals), expected, atol=0.3)

    def test_eigenvector_boundary_zeros_2d(self):
        nx = ny = 21
        shape = (nx, ny)
        dx = dy = np.pi / (nx - 1)
        L = Diff(0, dx) ** 2 + Diff(1, dy) ** 2

        bc = BoundaryConditions(shape)
        bc[0, :] = 0
        bc[-1, :] = 0
        bc[:, 0] = 0
        bc[:, -1] = 0

        vals, vecs = L.eigsh(shape, k=2, which='SM', bc=bc)

        # All boundary points should be zero
        assert_array_almost_equal(vecs[0, :, :], 0)
        assert_array_almost_equal(vecs[-1, :, :], 0)
        assert_array_almost_equal(vecs[:, 0, :], 0)
        assert_array_almost_equal(vecs[:, -1, :], 0)


class TestEigsWithoutBC:

    def test_periodic_laplacian_1d(self):
        """Periodic d^2/dx^2 on [0, 2*pi): eigenvalues are -n^2."""
        n = 64
        shape = (n,)
        x = np.linspace(0, 2 * np.pi, n, endpoint=False)
        dx = x[1] - x[0]

        L = Diff(0, dx, periodic=True) ** 2
        vals, vecs = L.eigsh(shape, k=5, which='SM')

        # Eigenvalues should include 0, -1, -1, -4, -4 (sorted: -4,-4,-1,-1,0)
        sorted_vals = np.sort(vals)
        np.testing.assert_allclose(sorted_vals[0], -4.0, atol=0.2)
        np.testing.assert_allclose(sorted_vals[1], -4.0, atol=0.2)
        np.testing.assert_allclose(sorted_vals[2], -1.0, atol=0.1)
        np.testing.assert_allclose(sorted_vals[3], -1.0, atol=0.1)
        np.testing.assert_allclose(sorted_vals[4], 0.0, atol=0.1)


class TestEigsVsEigshConsistency:

    def test_symmetric_operator(self):
        """eigs and eigsh should give the same eigenvalues for a symmetric operator."""
        n = 101
        shape = (n,)
        x = np.linspace(0, np.pi, n)
        dx = x[1] - x[0]
        L = Diff(0, dx) ** 2

        bc = BoundaryConditions(shape)
        bc[0] = 0
        bc[-1] = 0

        vals_eigs, _ = L.eigs(shape, k=3, which='SR', bc=bc)
        vals_eigsh, _ = L.eigsh(shape, k=3, which='SA', bc=bc)

        np.testing.assert_allclose(
            np.sort(vals_eigs.real), np.sort(vals_eigsh), atol=1e-6
        )


class TestGeneralizedEigenvalue:

    def test_identity_rhs_matches_standard(self):
        """L*u = lambda*I*u should equal L*u = lambda*u."""
        n = 101
        shape = (n,)
        x = np.linspace(0, np.pi, n)
        dx = x[1] - x[0]
        L = Diff(0, dx) ** 2

        bc = BoundaryConditions(shape)
        bc[0] = 0
        bc[-1] = 0

        vals_std, _ = L.eigsh(shape, k=3, which='SA', bc=bc)
        vals_gen, _ = L.eigsh(shape, k=3, which='SA', bc=bc, M=Identity())

        np.testing.assert_allclose(vals_std, vals_gen, atol=1e-6)
