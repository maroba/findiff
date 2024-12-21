from sympy import Add, IndexedBase, Matrix, Rational, Symbol, factorial, linsolve


class SymbolicMesh:
    """
    Represents the mesh on which to evaluate finite difference approximations.
    """

    def __init__(self, coord, equidistant=True):
        """Constructor.

        Parameters
        ----------
        coord: str
            A comma-separated string of coordinate names for the mesh,
            e.g. "x,y" or simply "x"

        equidistant: bool
            Flag indicating whether the mesh is equidistant.
        """
        assert isinstance(coord, str)

        self.equidistant = equidistant
        self._coord_names = [n.replace(" ", "") for n in coord.split(",")]
        self._coord = [IndexedBase(name) for name in self._coord_names]

    @property
    def ndims(self):
        """The number of dimensions of the mesh."""
        return len(self._coord)

    @property
    def coord(self):
        """
        Returns a tuple with the symbols for the coordinates.
        """
        return self._coord

    @property
    def spacing(self):
        """
        Returns a tuple with the spacing of the mesh along all axes.
        Only makes sense for equidistant grid. Raises exception in
        case of non-equidistant grids.
        """
        if self.equidistant:
            spacings = tuple(Symbol(f"\\Delta {x}") for x in self._coord_names)
            return spacings
        raise Exception("Non-equidistant mesh does not have spacing property.")

    @staticmethod
    def create_symbol(name):
        """
        Creates a *sympy* symbol of a given name which can carry as many
        indices as the mesh has dimensions.

        Parameters
        ----------
        name: str
            The name of the meshed symbol.

        Returns
        -------
        An index-carrying *sympy* symbol (IndexedBase).
        """
        return IndexedBase(name)


class SymbolicDiff:
    """
    A symbolic representation of the finite difference approximation
    of a partial derivative. Based on *sympy*.
    """

    def __init__(self, mesh, axis=0, degree=1):
        """Constructor

        Parameters
        ----------
        mesh: SymbolicMesh
            The symbolic grid on which to evaluate the derivative.
        axis: int
            The index of the axis with respect to which to differentiate.
        degree: int > 0
            The degree of the partial derivative.

        """
        self.mesh = mesh
        self.axis = axis
        self.degree = degree

    def __call__(self, f, at, offsets):
        if not isinstance(at, tuple) and not isinstance(at, list):
            at = [at]

        if self.mesh.ndims != len(at):
            raise ValueError("Index tuple must match the number of dimensions!")

        coefs = self._compute_coefficients(f, at, offsets)
        terms = []
        for coef, off in zip(coefs, offsets):
            inds = list(at)
            inds[self.axis] += off
            inds = tuple(inds)
            terms.append(coef * f[inds])

        return Add(*terms).simplify()

    def _compute_coefficients(self, f, at, offsets):

        n = len(offsets)
        # the first row always contains 1s:
        matrix = [[1] * n]

        def spac(off):
            """A helper function to get the spacing between grid points."""
            if self.mesh.equidistant:
                h = self.mesh.spacing[self.axis]
            else:
                x = self.mesh.coord[self.axis]
                h = x[at[self.axis] + off] - x[at[self.axis]]
            return h

        # build up the matrix incrementally:
        for i in range(1, n):
            ifac = Rational(1, factorial(i))
            row = [ifac * (off * spac(off)) ** i for off in offsets]
            matrix.append(row)

        # only the entry corresponding to the requested derivative degree
        # is non-zero:
        rhs = [0] * n
        rhs[self.degree] = 1

        # solve the equation system
        matrix = Matrix(matrix)
        rhs = Matrix(rhs)
        sol = linsolve((matrix, rhs))
        return list(sol)[0].simplify()
