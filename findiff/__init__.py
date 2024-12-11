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
- Version 1.0: Completely remodeled API (backward compatibility is maintained, though)
"""

__version__ = "0.11.1"


from .legacy import *
from .operators import Diff, Identity
from .pde import PDE, BoundaryConditions
from .compatible import Coef, Coefficient, FinDiff, Id
from .coefs import coefficients
from .vector import Gradient, Divergence, Curl, Laplacian
