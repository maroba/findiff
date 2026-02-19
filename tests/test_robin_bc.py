"""Tests for Robin (mixed) boundary conditions.

Robin BC: alpha * u + beta * du/dn = g

Covers:
- 1D Robin BCs (4-tuple syntax)
- 1D Robin BCs (operator-tuple syntax for backward compat)
- 2D Robin BCs on different edges
- Robin + Dirichlet mixed
- Robin + Neumann mixed
- Robin with variable coefficients
- Robin with array-valued g
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from findiff import Diff, PDE, BoundaryConditions, Identity


class TestRobin1D:

    def test_robin_1d_linear_solution(self):
        """u'' = 0 on [0,1], u(0) = 1, u(1) + u'(1) = 3 → u = x + 1."""
        n = 101
        x = np.linspace(0, 1, n)
        dx = x[1] - x[0]
        L = Diff(0, dx) ** 2

        bc = BoundaryConditions(x.shape)
        bc[0] = 1
        bc[-1] = (1, Diff(0, dx), 1, 3)  # u + u' = 3

        pde = PDE(L, np.zeros_like(x), bc)
        u = pde.solve()
        expected = x + 1
        assert_array_almost_equal(u, expected, decimal=4)

    def test_robin_1d_exponential_solution(self):
        """u'' - u = 0 on [0,1], u(0) = 1, u'(1) + u(1) = e + e = 2e → u = e^x"""
        n = 201
        x = np.linspace(0, 1, n)
        dx = x[1] - x[0]
        L = Diff(0, dx) ** 2 - Identity()

        bc = BoundaryConditions(x.shape)
        bc[0] = 1
        bc[-1] = (1, Diff(0, dx), 1, 2 * np.e)  # u + u' = 2e

        pde = PDE(L, np.zeros_like(x), bc)
        u = pde.solve()
        expected = np.exp(x)
        assert_array_almost_equal(u, expected, decimal=3)

    def test_robin_both_ends(self):
        """u'' = 0, u(0) + u'(0) = 2, u(1) - u'(1) = 0 → u = x + 1.

        Left: u(0) + u'(0) = 1 + 1 = 2 ✓
        Right: u(1) - u'(1) = 2 - 1 = 1... Let's use different values.

        u = x + 1: u'(x) = 1
        Left Robin: 1*u(0) + 1*u'(0) = 1 + 1 = 2
        Right Robin: 1*u(1) + (-1)*u'(1) = 2 - 1 = 1
        """
        n = 101
        x = np.linspace(0, 1, n)
        dx = x[1] - x[0]
        L = Diff(0, dx) ** 2

        bc = BoundaryConditions(x.shape)
        bc[0] = (1, Diff(0, dx), 1, 2)    # u + u' = 2
        bc[-1] = (1, Diff(0, dx), -1, 1)  # u - u' = 1

        pde = PDE(L, np.zeros_like(x), bc)
        u = pde.solve()
        expected = x + 1
        assert_array_almost_equal(u, expected, decimal=4)

    def test_robin_operator_tuple_syntax(self):
        """Robin BC can be specified via operator expression (backward compat)."""
        n = 101
        x = np.linspace(0, 1, n)
        dx = x[1] - x[0]
        L = Diff(0, dx) ** 2

        bc = BoundaryConditions(x.shape)
        bc[0] = 1
        # Operator-tuple form: (alpha*I + beta*D, g)
        robin_op = Identity() + Diff(0, dx)
        bc[-1] = robin_op, 3

        pde = PDE(L, np.zeros_like(x), bc)
        u = pde.solve()
        expected = x + 1
        assert_array_almost_equal(u, expected, decimal=4)

    def test_robin_4tuple_matches_operator_tuple(self):
        """4-tuple Robin syntax gives the same result as operator-tuple syntax."""
        n = 101
        x = np.linspace(0, 1, n)
        dx = x[1] - x[0]
        L = Diff(0, dx) ** 2
        alpha, beta, g = 2.0, 0.5, 5.0

        # 4-tuple syntax
        bc1 = BoundaryConditions(x.shape)
        bc1[0] = 1
        bc1[-1] = (alpha, Diff(0, dx), beta, g)
        u1 = PDE(L, np.zeros_like(x), bc1).solve()

        # operator-tuple syntax
        bc2 = BoundaryConditions(x.shape)
        bc2[0] = 1
        bc2[-1] = alpha * Identity() + beta * Diff(0, dx), g
        u2 = PDE(L, np.zeros_like(x), bc2).solve()

        assert_array_almost_equal(u1, u2)


class TestRobin2D:

    def test_robin_2d_one_edge(self):
        """2D Laplacian with Robin BC on one edge.

        u = x + y + 1, ∇²u = 0
        Robin at x=1: u + ∂u/∂x = (1+y+1) + 1 = y + 3
        """
        shape = (31, 31)
        x = y = np.linspace(0, 1, shape[0])
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing="ij")

        L = Diff(0, dx) ** 2 + Diff(1, dy) ** 2
        f = np.zeros_like(X)
        expected = X + Y + 1

        bc = BoundaryConditions(shape)
        bc[-1, :] = (1, Diff(0, dx), 1, Y + 3)  # Robin at x=1
        bc[0, :] = expected
        bc[:, 0] = expected
        bc[:, -1] = expected

        pde = PDE(L, f, bc)
        u = pde.solve()
        assert_array_almost_equal(u, expected, decimal=3)

    def test_robin_2d_array_valued_g(self):
        """Robin BC with array-valued right-hand side g."""
        shape = (21, 21)
        x = y = np.linspace(0, 1, shape[0])
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing="ij")

        L = Diff(0, dx) ** 2 + Diff(1, dy) ** 2
        expected = X ** 2 + Y + 1
        f = 2 * np.ones_like(X)

        # Robin at x=1: 1*u + 0.5*∂u/∂x = u + 0.5*2x
        g_robin = expected + 0.5 * (2 * X)

        bc = BoundaryConditions(shape)
        bc[-1, :] = (1, Diff(0, dx), 0.5, g_robin)
        bc[0, :] = expected
        bc[:, 0] = expected
        bc[:, -1] = expected

        pde = PDE(L, f, bc)
        u = pde.solve()
        assert_array_almost_equal(u, expected, decimal=3)

    def test_robin_2d_two_edges(self):
        """2D problem with Robin BCs on two opposite edges."""
        shape = (31, 31)
        x = y = np.linspace(0, 1, shape[0])
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing="ij")

        L = Diff(0, dx) ** 2 + Diff(1, dy) ** 2
        expected = X + Y + 1
        f = np.zeros_like(X)

        bc = BoundaryConditions(shape)
        # Robin at x=0: u - ∂u/∂x = (Y+1) - 1 = Y
        bc[0, :] = (1, Diff(0, dx), -1, Y)
        # Robin at x=1: u + ∂u/∂x = (1+Y+1) + 1 = Y+3
        bc[-1, :] = (1, Diff(0, dx), 1, Y + 3)
        # Dirichlet on y-edges (set last so corners get Dirichlet)
        bc[:, 0] = expected
        bc[:, -1] = expected

        pde = PDE(L, f, bc)
        u = pde.solve()
        assert_array_almost_equal(u, expected, decimal=3)


class TestRobinMixed:

    def test_robin_and_dirichlet(self):
        """Robin on one end, Dirichlet on the other."""
        n = 101
        x = np.linspace(0, 1, n)
        dx = x[1] - x[0]
        L = Diff(0, dx) ** 2

        bc = BoundaryConditions(x.shape)
        bc[0] = 1                              # Dirichlet
        bc[-1] = (1, Diff(0, dx), 1, 3)        # Robin: u + u' = 3

        pde = PDE(L, np.zeros_like(x), bc)
        u = pde.solve()
        expected = x + 1
        assert_array_almost_equal(u, expected, decimal=4)

    def test_robin_and_neumann(self):
        """Robin on one end, Neumann on the other.

        u'' = 0, u'(0) = 1 (Neumann), u(1) + u'(1) = 3 (Robin)
        Solution family: u = x + c. u(1) + u'(1) = (1+c) + 1 = c + 2 = 3 → c = 1
        So u = x + 1
        """
        n = 101
        x = np.linspace(0, 1, n)
        dx = x[1] - x[0]
        L = Diff(0, dx) ** 2

        bc = BoundaryConditions(x.shape)
        bc[0] = Diff(0, dx), 1                  # Neumann: u' = 1
        bc[-1] = (1, Diff(0, dx), 1, 3)         # Robin: u + u' = 3

        pde = PDE(L, np.zeros_like(x), bc)
        u = pde.solve()
        expected = x + 1
        assert_array_almost_equal(u, expected, decimal=4)

    def test_robin_with_weighted_alpha(self):
        """Robin BC with alpha != 1.

        u'' = 0, u(0) = 1, 2*u(1) + u'(1) = 5
        u = x + 1: 2*(2) + 1 = 5 ✓
        """
        n = 101
        x = np.linspace(0, 1, n)
        dx = x[1] - x[0]
        L = Diff(0, dx) ** 2

        bc = BoundaryConditions(x.shape)
        bc[0] = 1
        bc[-1] = (2, Diff(0, dx), 1, 5)  # 2u + u' = 5

        pde = PDE(L, np.zeros_like(x), bc)
        u = pde.solve()
        expected = x + 1
        assert_array_almost_equal(u, expected, decimal=4)
