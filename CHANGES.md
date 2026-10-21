# Change Log

## Unreleased

- Compact schemes: much faster application (LU factorization is computed once and
  reused) and `matrix()` no longer builds a dense intermediate matrix, which made
  compact schemes unusable in `PDE` / `TimeDependentPDE` on medium-sized 2D grids
- Compact schemes: one-sided boundary stencils on non-periodic grids now reach the
  interior accuracy order (previously one to two orders lower).
  **Note:** results at the boundary points of non-periodic compact schemes change
  (they become more accurate).
- Compact schemes: boundary fallback to explicit differences now only catches
  `LinAlgError` and emits a `RuntimeWarning` instead of silently swallowing all errors
- Compact schemes: clear error message if the grid is too small for the stencil

## Version 0.14.x

- Time-dependent PDEs via Method of Lines (Forward Euler, RK4, Backward Euler, Crank-Nicolson)
- GPU / JAX / CuPy backend support for operator application
- Truncation error estimation via accuracy order comparison
- Support for non-uniform grids in Laplacian
- Iterative solvers for PDEs (CG, GMRES, BiCGSTAB, LGMRES, MINRES) with preconditioner support
- Eigenvalue problem support (eigs and eigsh methods)
- Robin (mixed) boundary conditions for PDEs
- Fix order-of-accuracy loss for even-order derivatives on non-uniform grids (central stencil was undersized).
  **Note:** results for even-order derivatives on non-uniform grids (`coords=`) change (they become more
  accurate), stencils are one point wider, and the minimum grid size for these derivatives grows by one.
- Fix `PDE.solve()` overwriting the caller's `rhs` and truncating boundary values for integer arrays
- Fix `TimeDependentPDE` not applying Dirichlet values to the initial state
- Fix crash when passing a 0-d array (e.g. JAX/NumPy scalar) as grid spacing

## Version 0.13.x

- Implements compact difference schemes

## Version 0.12.x

- Implements periodic boundary conditions

## Version 0.11.x

- More comfortable API. (Old API still available)

## Version 0.10.x

- Create symbolic representations of finite difference schemes

## Version 0.9.x

- Generate differential operators for generic stencils

## Version 0.8.x

- Formulate and solve partial differential equations

## Version 0.7.*

- Generation of stencils for given FinDiff object
- Generation of matrix representation for arbitrary linear differential operator
