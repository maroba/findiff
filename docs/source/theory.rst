Some Theory
===========

As the name *findiff* suggests, the package uses finite difference
schemes to approximate differential operators numerically. In this
section, we describe the method in some detail.

Notation
--------

In this section, we are talking about functions on equidistant grids.
So in 1D, instead of a continuous variable :math:`x`, we have a set
of grid points

.. math::

    x_k = a + k \Delta x

for some real number :math:`a` and grid spacing :math:`\Delta x`. In many
dimensions, say 3, we have

.. math::

    x_{ijk} = \left(
    \begin{matrix}
         x_i \\
         y_j \\
         z_k \\
    \end{matrix}
    \right) =
     \left(
    \begin{matrix}
         a_x + i \Delta x \\
         a_y + j \Delta y \\
         a_z + k \Delta z \\
    \end{matrix}
    \right)

For a function *f* given on a grid, we write

.. math::

    f_{ijk} = f(x_{ijk})

The generalization to *N* dimensions is straight forward.


The 1D Case
------------

Say we want to calculate the *n*-th derivative :math:`\frac{d^n f}{dx^n}` of
a function of a single variable and let the function be given on an equidistant
grid. The basic idea behind finite difference is to approximate the true
derivative at some point :math:`x_k` by a linear combination of the function
values around :math:`x_k`.

.. math::

    \left(\frac{d^n f}{dx^n}\right)_k \approx \sum_{i \in A} c_{i} f_{k+i}



The Rest
--------

*FinDiff* objects make no assumption on the dimensionality of the
space.