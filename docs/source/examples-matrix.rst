Matrix Representations
======================

The ``FinDiff`` objects are designed to easily and quickly apply partial derivatives
to given *numpy* arrays. However, sometimes it is useful to represent
the linear differential operator as a matrix. Consider for instance the
stationary Schrödinger equation in 3D:

.. math::

    \frac{\partial^2}{\partial x^2}\psi + \frac{\partial^2}{\partial y^2}\psi + \frac{\partial^2}{\partial z^2}\psi + V(x, y, z)\psi = E \psi

This is an eigenvalue problem for the differential operator

.. math::

    H = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2} + \frac{\partial^2}{\partial z^2} + V(x, y, z)

The Laplacian can be expressed as a ``FinDiff`` object, e.g. as

.. code::

   >>> laplace = FinDiff(0, dx, 2) + FinDiff(1, dy, 2) + FinDiff(2, dz, 2)

Then we can convert the resulting ``FinDiff`` object into a matrix
representation by calling its ```matrix`` method:

.. code::

   >>> mat = laplace.matrix(shape)

where ``shape`` is a tuple describing the shape (number of grid points)
of the given grid.

The Schrödinger equation can then be solved with the eigenvalue solver
of ``scipy``:

.. code::

   >>> hamiltonian = laplace.matrix(shape) + V.reshape(-1)
   >>> scipy.sparse.linalg.eigs(hamiltonian, k=6, which='SR')

This returns the ``k=6`` eigenvalues and eigenfunctions of the Hamiltonian
with the smallest real value (``'SR'``).
