Schrödinger Equation
====================

The ``FinDiff`` objects are designed to easily and quickly apply partial derivatives
to given *numpy* arrays. However, sometimes it is useful to represent
the linear differential operator as a matrix. Consider for instance the
stationary Schrödinger equation in 3D:

.. math::

    -\frac{1}{2}\left(\frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2} + \frac{\partial^2}{\partial z^2}\right) \psi + V(x, y, z)\psi = E \psi

This is an eigenvalue problem for the differential operator

.. math::

    H = -\frac{1}{2}\left(\frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2} + \frac{\partial^2}{\partial z^2}\right) \psi + V(x, y, z)

The Laplacian can be expressed as a ``Diff`` object, e.g. as

.. code::

   >>> laplace = Diff(0, dx)**2 + Diff(1, dy)**2 + Diff(2, dz)**2

Then we can convert the resulting ``Diff`` object into a matrix
representation by calling its ``matrix`` method:

.. code::

   >>> mat = laplace.matrix(shape)

where ``shape`` is a tuple describing the shape (number of grid points)
of the given grid.

The Schrödinger equation can then be solved with the eigenvalue solver
of ``scipy``:

.. code::

   >>> hamiltonian = - laplace.matrix(shape) + V.reshape(-1)
   >>> scipy.sparse.linalg.eigs(hamiltonian, k=6, which='SR')

This returns the ``k=6`` eigenvalues and eigenfunctions of the Hamiltonian
with the smallest real value (``'SR'``).
