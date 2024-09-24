from setuptools import setup, find_packages


setup(
    name="findiff",
    packages=find_packages(where=".", include=["findiff*"]),
    long_description="""A Python package for finite difference derivatives in any number of dimensions.

    Features:

        * Differentiate arrays of any number of dimensions along any axis
        * Partial derivatives of any desired order
        * Accuracy order can be specified
        * Accurate treatment of grid boundary
        * Includes standard operators from vector calculus like gradient, divergence and curl
        * Can handle uniform and non-uniform grids
        * Can handle arbitrary linear combinations of derivatives with constant and variable coefficients
        * Fully vectorized for speed
        * Calculate raw finite difference coefficients for any order and accuracy for uniform and non-uniform grids
        * _New in version 0.7:_ Generate matrix representations of arbitrary linear differential operators
        * _New in version 0.8:_ Solve partial differential equations with Dirichlet or Neumann boundary conditions
        * _New in version 0.9:_ Generate differential operators for generic stencils
        * _New in version 0.10:_ Create symbolic representations of finite difference schemes
    """,
)
