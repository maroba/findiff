# Change Log

## Unreleased

- `TimeDependentPDE`: implicit methods (`backward-euler`, `crank-nicolson`) are much faster
  (~70-180x per step on 50x50 to 100x100 grids). Boundary conditions are injected once with
  sparse masks instead of converting the system matrix to LIL format in every step, the LU
  factorization is reused, and time steps differing only by round-off (e.g. from
  `np.linspace`) no longer trigger a rebuild of the system.

## Version 0.15.x

- Compact schemes with spectral-like resolution (Lele, 1992): `Diff(..., compact=5, acc=4,
  spectral=True)` or `CompactScheme.spectral(deriv, num_left, num_right, acc)`. Only order `acc`
  is imposed; the remaining coefficients minimize the modified wavenumber error. New helpers
  `CompactScheme.modified_wavenumber()` and `resolving_efficiency()`; `CompactScheme` accepts
  explicit RHS `coefficients`.
- Compact schemes on non-uniform grids: `Diff(0, coords, compact=3, acc=6)` (previously
  `NotImplementedError`). LHS alphas of the uniform scheme are kept, RHS coefficients are solved
  per row from the actual spacings; works on arbitrary (also strongly clustered) grids.
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
- Compact schemes: `compact=..., acc=...` now determines the LHS alphas and RHS coefficients
  jointly (as in Lele, 1992) instead of using fixed heuristic alphas, and picks the narrowest
  RHS stencil reaching `acc`. This yields the classical schemes (Padé, Lele's tridiagonal and
  pentadiagonal schemes); `compact=5` and `compact=7` now actually pay off (e.g. `compact=5,
  acc=10` needs 7 instead of 11 RHS points and is ~85x more accurate).
  **Note:** results change for all `compact=` settings except `compact=3, acc=6`.
  `acc` is now the minimum order, as for explicit differences: `compact=3, acc=4` gives the
  4th-order Padé scheme (previously 6th order by accident), while the default `acc=2` now
  gives 4th instead of 2nd order.
- Fix accuracy order detection for high-order schemes (truncation error was compared with an
  absolute tolerance, causing an `OverflowError` from about order 12 on)

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
