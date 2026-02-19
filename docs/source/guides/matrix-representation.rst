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


Example: Schrodinger Equation
------------------------------

The stationary Schrodinger equation in 3D is an eigenvalue problem:

.. math::

    -\frac{1}{2}\left(\frac{\partial^2}{\partial x^2}
    + \frac{\partial^2}{\partial y^2}
    + \frac{\partial^2}{\partial z^2}\right) \psi
    + V(x, y, z)\psi = E \psi

Build the Hamiltonian matrix and solve:

.. code:: python

    import scipy.sparse.linalg

    laplace = Diff(0, dx)**2 + Diff(1, dy)**2 + Diff(2, dz)**2

    hamiltonian = -0.5 * laplace.matrix(shape) + V.reshape(-1)
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(
        hamiltonian, k=6, which='SR'
    )

This returns the 6 eigenvalues with the smallest real part and the
corresponding eigenfunctions.
