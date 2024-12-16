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
- Generate matrix representations of arbitrary linear differential operators
- Solve partial differential equations with Dirichlet or Neumann boundary conditions
- Generate differential operators for generic stencils
- Create symbolic representations of finite difference schemes
- Version 0.11.*: Completely remodeled API (backward compatibility is maintained, though)
- Version 0.12.*: Periodic boundary conditions for differential operators and PDEs
"""

__version__ = "0.12.0"


from .operators import Diff, Identity
from .pde import PDE, BoundaryConditions
from .compatible import Coef, Coefficient, FinDiff, Id
from .coefs import coefficients
from .vector import Gradient, Divergence, Curl, Laplacian
