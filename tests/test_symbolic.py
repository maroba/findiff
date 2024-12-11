import unittest

from sympy import IndexedBase, Symbol, symbols, latex

from findiff.symbolic import SymbolicMesh, SymbolicDiff


class TestSymbolicMesh(unittest.TestCase):

    def test_parse_symbolic_mesh(self):
        # 1D
        mesh = SymbolicMesh(coord="x", equidistant=True)
        (x,) = mesh.coord
        (dx,) = mesh.spacing

        self.assertEqual(IndexedBase, type(x))
        self.assertEqual(Symbol, type(dx))

        # 2D
        mesh = SymbolicMesh(coord="x,y", equidistant=True)
        x, y = mesh.coord
        dx, dy = mesh.spacing

        self.assertEqual(IndexedBase, type(x))
        self.assertEqual(IndexedBase, type(y))
        self.assertEqual(Symbol, type(dx))
        self.assertEqual(Symbol, type(dy))

        # ignores whitespace
        mesh = SymbolicMesh(coord="x,  y", equidistant=True)
        x, y = mesh.coord

        self.assertEqual("x", str(x))
        self.assertEqual("y", str(y))

    def test_create_symbol(self):
        # defaults
        mesh = SymbolicMesh(coord="x", equidistant=True)
        actual = mesh.create_symbol("u")
        expected = IndexedBase("u")
        self.assertEqual(latex(actual), latex(expected))

        # both indices down
        mesh = SymbolicMesh(coord="x,y", equidistant=True)
        n, m = symbols("n, m")
        actual = latex(mesh.create_symbol("u")[n, m])
        # expected = "u{}_{n}{}_{m}"
        expected = "{u}_{n,m}"
        self.assertEqual(latex(actual), latex(expected))

        # both indices up
        # mesh = SymbolicMesh(coord="x,y", equidistant=True)
        # n, m = symbols("n, m")
        # u = mesh.create_symbol("u", pos="u,u")
        # actual = latex(u[n, m])
        # expected = "u{}^{n}{}^{m}"
        # self.assertEqual(actual, expected)


class TestDiff(unittest.TestCase):

    def test_init(self):
        mesh = SymbolicMesh("x")
        d = SymbolicDiff(mesh)

        self.assertEqual(d.axis, 0)
        self.assertEqual(d.degree, 1)
        self.assertEqual(id(mesh), id(d.mesh))

    def test_call(self):
        # 1D
        mesh = SymbolicMesh("x")
        u = mesh.create_symbol("u")
        d = SymbolicDiff(mesh)
        n = Symbol("n")

        actual = d(u, at=n, offsets=[-1, 0, 1])

        expected = (u[n + 1] - u[n - 1]) / (2 * mesh.spacing[0])

        self.assertEqual(0, (expected - actual).simplify())

        # 2D
        mesh = SymbolicMesh("x, y")
        u = mesh.create_symbol("u")
        d = SymbolicDiff(mesh, axis=1)
        n, m = symbols("n, m")

        actual = d(u, at=(n, m), offsets=[-1, 0, 1])

        expected = (u[n, m + 1] - u[n, m - 1]) / (2 * mesh.spacing[1])

        self.assertEqual(0, (expected - actual).simplify())


if __name__ == "__main__":
    unittest.main()
