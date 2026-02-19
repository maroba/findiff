=======================
Matrix Representation
=======================

``Diff`` objects can be converted to sparse matrix representations.
This is useful for eigenvalue problems or when you need to solve
linear systems involving differential operators.


Basic usage
-----------

.. code:: python

    import numpy as np
    from findiff import Diff

    x = np.linspace(0, 6, 7)
    d2_dx2 = Diff(0, x[1] - x[0]) ** 2

    mat = d2_dx2.matrix(x.shape)
    print(mat.toarray())

Output:

.. code:: python

    [[ 2. -5.  4. -1.  0.  0.  0.]
     [ 1. -2.  1.  0.  0.  0.  0.]
     [ 0.  1. -2.  1.  0.  0.  0.]
     [ 0.  0.  1. -2.  1.  0.  0.]
     [ 0.  0.  0.  1. -2.  1.  0.]
     [ 0.  0.  0.  0.  1. -2.  1.]
     [ 0.  0.  0. -1.  4. -5.  2.]]

The returned matrix is a ``scipy.sparse`` matrix.


Eigenvalue Problems
--------------------

Differential operators have built-in ``eigs`` and ``eigsh`` methods for
solving eigenvalue problems directly.

**Example: 1D quantum harmonic oscillator**

.. math::

    -\frac{1}{2}\frac{d^2\psi}{dx^2} + \frac{1}{2}x^2\psi = E\,\psi

The exact eigenvalues are :math:`E_n = n + \tfrac{1}{2}`.

.. code:: python

    from findiff import Diff, Identity, BoundaryConditions

    n = 300
    x = np.linspace(-8, 8, n)
    dx = x[1] - x[0]

    H = -0.5 * Diff(0, dx)**2 + 0.5 * x**2 * Identity()

    bc = BoundaryConditions((n,))
    bc[0] = 0
    bc[-1] = 0

    eigenvalues, eigenvectors = H.eigsh((n,), k=6, which='SM', bc=bc)
    # eigenvalues ≈ [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]

The ``eigsh`` method is for symmetric operators (Laplacian, Hamiltonian).
Use ``eigs`` for general (non-symmetric) operators.

**Key parameters:**

- ``k``: number of eigenvalues to compute
- ``which``: which eigenvalues — ``'SM'`` (smallest magnitude),
  ``'SA'`` (smallest algebraic), ``'LM'`` (largest magnitude)
- ``bc``: a ``BoundaryConditions`` object; boundary DOFs are eliminated
  (homogeneous Dirichlet)
- ``sigma``: shift for shift-invert mode (finds eigenvalues near ``sigma``)
- ``M``: another operator for generalized problems :math:`L\psi = \lambda M\psi`

Eigenvectors are returned with shape ``(*grid_shape, k)``, so
``eigenvectors[..., i]`` is the *i*-th eigenvector on the grid.

**Example: 3D Schrodinger equation**

.. math::

    -\frac{1}{2}\nabla^2 \psi + V\,\psi = E\,\psi

.. code:: python

    laplace = Diff(0, dx)**2 + Diff(1, dy)**2 + Diff(2, dz)**2
    H = -0.5 * laplace + V * Identity()

    bc = BoundaryConditions(shape)
    bc[0, :, :] = 0
    bc[-1, :, :] = 0
    bc[:, 0, :] = 0
    bc[:, -1, :] = 0
    bc[:, :, 0] = 0
    bc[:, :, -1] = 0

    eigenvalues, eigenvectors = H.eigsh(shape, k=6, which='SM', bc=bc)
