"""
findiff is a Python package for finite difference numerical derivatives
and partial differential equations in any number of dimensions.

Features:

- Differentiate arrays of any number of dimensions along any axis with any desired accuracy order
- Accurate treatment of grid boundary
- Includes standard operators from vector calculus like gradient, divergence and curl
- Can handle uniform and non-uniform grids
- Can handle arbitrary linear combinations of derivatives with constant and variable coefficients
- Fully vectorized for speed
- Calculate raw finite difference coefficients for any order and accuracy for uniform and non-uniform grids
- New in version 0.7: Generate matrix representations of arbitrary linear differential operators
- New in version 0.8: Solve partial differential equations with Dirichlet or Neumann boundary conditions
- New in version 0.9: Generate differential operators for generic stencils
- New in version 0.10: Create symbolic representations of finite difference schemes
"""

__version__ = "0.10.2"

from .legacy.coefs import coefficients
from .legacy.operators import FinDiff, Coef, Identity, Coefficient
from .legacy.pde import PDE, BoundaryConditions
from .legacy.symbolic import SymbolicMesh, SymbolicDiff
from .legacy.vector import Gradient, Divergence, Curl, Laplacian
