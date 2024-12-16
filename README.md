# <img src="docs/frontpage/findiff_logo.png" width="100px"> findiff

[![PyPI version](https://badge.fury.io/py/findiff.svg)](https://img.shields.io/pypi/v/findiff.png?style=flat-square&color=brightgreen)
![build](https://github.com/maroba/findiff/actions/workflows/check.yml/badge.svg)
![Coverage](https://img.shields.io/codecov/c/github/maroba/findiff/master.svg)
[![Doc Status](https://readthedocs.org/projects/findiff/badge/?version=latest)](https://findiff.readthedocs.io/en/latest/index.html)
[![PyPI downloads](https://img.shields.io/pypi/dm/findiff.svg)]()
[![Downloads](https://static.pepy.tech/personalized-badge/findiff?period=total&units=international_system&left_color=black&right_color=blue&left_text=Downloads)](https://pepy.tech/project/findiff)

A Python package for finite difference numerical derivatives and partial differential equations in
any number of dimensions.

## Main Features

* Differentiate arrays of any number of dimensions along any axis with any desired accuracy order
* Accurate treatment of grid boundary
* Can handle arbitrary linear combinations of derivatives with constant and variable coefficients
* Fully vectorized for speed
* Matrix representations of arbitrary linear differential operators
* Solve partial differential equations with Dirichlet or Neumann boundary conditions
* Symbolic representation of finite difference schemes
* **New in version 0.11**: More comfortable API (keeping the old API available)
* **New in version 0.12**: Periodic boundary conditions for differential operators and PDEs.

## Installation

```
pip install --upgrade findiff
```

## Documentation and Examples

You can find the documentation of the code including examples of application at https://findiff.readthedocs.io/en/stable/.

## Taking Derivatives

*findiff* allows to easily define derivative operators that you can apply to *numpy* arrays of 
any dimension. 

Consider the simple 1D case of a equidistant grid
with a first derivative $\displaystyle \frac{\partial}{\partial x}$ along the only axis (0):

```python
import numpy as np
from findiff import Diff

# define the grid:
x = np.linspace(0, 1, 100)

# the array to differentiate:
f = np.sin(x)  # as an example

# Define the derivative:
d_dx = Diff(0, x[1] - x[0])

# Apply it:
df_dx = d_dx(f) 
```

Similarly, you can define partial derivatives along other axes, for example, if $z$ is the 2-axis, we can write
$\frac{\partial}{\partial z}$ as:

```python
Diff(2, dz)
```

`Diff` always creates a first derivative. For higher derivatives, you simply exponentiate them, for example for $\frac{\partial^2}{\partial_x^2}$

```
d2_dx2 = Diff(0, dx)**2
```

and apply it as before.

You can also define more general differential operators intuitively, like

$$
2x \frac{\partial^3}{\partial x^2 \partial z} + 3 \sin(y)z^2 \frac{\partial^3}{\partial x \partial y^2}
$$


which can be written as

```python
# define the operator
diff_op = 2 * X * Diff(0)**2 * Diff(2) + 3 * sin(Y) * Z**2 * Diff(0) * Diff(1)**2

# set the grid you use (equidistant here)
diff_op.set_grid({0: dx, 1: dy, 2: dz})

# apply the operator
result = diff_op(f)
```

where `X, Y, Z` are *numpy* arrays with meshed grid points. Here you see that you can also define your grid
lazily.

Of course, standard operators from vector calculus like gradient, divergence and curl are also available
as shortcuts.

If one or more axis of your grid are periodic, you can specify that when defining the derivative or later
when setting the grid. For example:

```python
d_dx = Diff(0, dx, periodic=True)

# or later
d_dx = Diff(0)
d_dx.set_grid({0: {"h": dx, "periodic": True}})
```

More examples can be found [here](https://findiff.readthedocs.io/en/latest/source/examples.html) and in [this blog](https://medium.com/p/7e54132a73a3).

### Accuracy Control

When constructing an instance of `Diff`, you can request the desired accuracy
order by setting the keyword argument `acc`. For example:

```python
d_dx = Diff(0, dy, acc=4)
df_dx = d2_dx2(f)
```

Alternatively, you can also split operator definition and configuration:

```python
d_dx = Diff(0, dx)
d_dx.set_accuracy(2)
df_dx = d2_dx2(f)
```

which comes in handy if you have a complicated expression of differential operators, because then you
can specify it on the whole expression and it will be passed down to all basic operators.

If not specified, second order accuracy will be taken by default.

## Finite Difference Coefficients

Sometimes you may want to have the raw finite difference coefficients.
These can be obtained for __any__ derivative and accuracy order
using `findiff.coefficients(deriv, acc)`. For instance,

```python
import findiff
coefs = findiff.coefficients(deriv=3, acc=4, symbolic=True)
```

gives

```
{'backward': {'coefficients': [15/8, -13, 307/8, -62, 461/8, -29, 49/8],
              'offsets': [-6, -5, -4, -3, -2, -1, 0]},
 'center': {'coefficients': [1/8, -1, 13/8, 0, -13/8, 1, -1/8],
            'offsets': [-3, -2, -1, 0, 1, 2, 3]},
 'forward': {'coefficients': [-49/8, 29, -461/8, 62, -307/8, 13, -15/8],
             'offsets': [0, 1, 2, 3, 4, 5, 6]}}
```

If you want to specify the detailed offsets instead of the
accuracy order, you can do this by setting the offset keyword
argument:

```python
import findiff
coefs = findiff.coefficients(deriv=2, offsets=[-2, 1, 0, 2, 3, 4, 7], symbolic=True)
```

The resulting accuracy order is computed and part of the output:

```
{'coefficients': [187/1620, -122/27, 9/7, 103/20, -13/5, 31/54, -19/2835], 
 'offsets': [-2, 1, 0, 2, 3, 4, 7], 
 'accuracy': 5}
```

## Matrix Representation

For a given differential operator, you can get the matrix representation
using the `matrix(shape)` method, e.g. for a small 1D grid of 10 points:

```python
d2_dx2 = Diff(0, dx)**2
mat = d2_dx2.matrix((10,))  # this method returns a scipy sparse matrix
print(mat.toarray())
```

has the output

```
[[ 2. -5.  4. -1.  0.  0.  0.]
 [ 1. -2.  1.  0.  0.  0.  0.]
 [ 0.  1. -2.  1.  0.  0.  0.]
 [ 0.  0.  1. -2.  1.  0.  0.]
 [ 0.  0.  0.  1. -2.  1.  0.]
 [ 0.  0.  0.  0.  1. -2.  1.]
 [ 0.  0.  0. -1.  4. -5.  2.]]
```

If you have periodic boundary conditions, the matrix looks like that:

```python
d2_dx2 = Diff(0, dx, periodic=True)**2
mat = d2_dx2.matrix((10,))  # this method returns a scipy sparse matrix
print(mat.toarray())
```

```
[[-2.  1.  0.  0.  0.  0.  1.]
 [ 1. -2.  1.  0.  0.  0.  0.]
 [ 0.  1. -2.  1.  0.  0.  0.]
 [ 0.  0.  1. -2.  1.  0.  0.]
 [ 0.  0.  0.  1. -2.  1.  0.]
 [ 0.  0.  0.  0.  1. -2.  1.]
 [ 1.  0.  0.  0.  0.  1. -2.]]
```

## Stencils

*findiff* uses standard stencils (patterns of grid points) to evaluate the derivative.
However, you can design your own stencil. A picture says more than a thousand words, so
look at the following example for a standard second order accurate stencil for the
2D Laplacian $\displaystyle \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}$:

<img src="docs/frontpage/laplace2d.png" width="400">

This can be reproduced by *findiff* writing

```python
offsets = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
stencil = Stencil(offsets, partials={(2, 0): 1, (0, 2): 1}, spacings=(1, 1))
```

The attribute `stencil.values` contains the coefficients

```
{(0, 0): -4.0, (1, 0): 1.0, (-1, 0): 1.0, (0, 1): 1.0, (0, -1): 1.0}
```

Now for a some more exotic stencil. Consider this one:

<img src="docs/frontpage/laplace2d-x.png" width="400">

With *findiff* you can get it easily:

```python
offsets = [(0, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
stencil = Stencil(offsets, partials={(2, 0): 1, (0, 2): 1}, spacings=(1, 1))
stencil.values
```

which returns

```
{(0, 0): -2.0, (1, 1): 0.5, (-1, -1): 0.5, (1, -1): 0.5, (-1, 1): 0.5}
```

## Symbolic Representations

As of version 0.10, findiff can also provide a symbolic representation of finite difference schemes suitable for using in conjunction with sympy. The main use case is to facilitate deriving your own iteration schemes.

```python
from findiff import SymbolicMesh, SymbolicDiff

mesh = SymbolicMesh("x, y")
u = mesh.create_symbol("u")
d2_dx2, d2_dy2 = [SymbolicDiff(mesh, axis=k, degree=2) for k in range(2)]

(
    d2_dx2(u, at=(m, n), offsets=(-1, 0, 1)) + 
    d2_dy2(u, at=(m, n), offsets=(-1, 0, 1))
)
```

Outputs:

$$
\frac{u_{m,n + 1} + u_{m,n - 1} - 2 u_{m,n}}{\Delta y^2}  + \frac{u_{m + 1,n} + u_{m - 1,n} - 2 u_{m,n}}{\Delta x^2}
$$

Also see the [example notebook](examples/symbolic.ipynb).

## Partial Differential Equations

_findiff_ can be used to easily formulate and solve partial differential equation problems

$$
\mathcal{L}u(\vec{x}) = f(\vec{x})
$$

where $\mathcal{L}$ is a general linear differential operator.

In order to obtain a unique solution,  Dirichlet, Neumann or more general boundary conditions
can be applied.

### Boundary Value Problems

#### Example 1: 1D forced harmonic oscillator with friction

Find the solution of

$$
\left( \frac{d^2}{dt^2} - \alpha \frac{d}{dt} + \omega^2 \right)u(t) = \sin{(2t)}
$$

subject to the (Dirichlet) boundary conditions

$$
u(0) = 0, \hspace{1em} u(10) = 1
$$

```python
from findiff import Diff, Id, PDE

shape = (300, )
t = numpy.linspace(0, 10, shape[0])
dt = t[1]-t[0]

L = Diff(0, dt)**2 - Diff(0, dt) + 5 * Id()
f = numpy.cos(2*t)

bc = BoundaryConditions(shape)
bc[0] = 0
bc[-1] = 1

pde = PDE(L, f, bc)
u = pde.solve()
```

Result:

<p align="center">
<img src="docs/frontpage/ho_bvp.jpg" alt="ResultHOBVP" height="300"/>
</p>

#### Example 2: 2D heat conduction

A plate with temperature profile given on one edge and zero heat flux across the other
edges, i.e.

$$
\left( \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2} \right) u(x,y) = f(x,y)
$$

with Dirichlet boundary condition

$$
\begin{align*}
u(x,0) &= 300 \\
u(1,y) &= 300 - 200y
\end{align*}
$$

and Neumann boundary conditions

$$
\begin{align*}
\frac{\partial u}{\partial x} &= 0, & \text{ for } x = 0 \\
\frac{\partial u}{\partial y} &= 0, & \text{ for } y = 0
\end{align*}
$$

```python
shape = (100, 100)
x, y = np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1])
dx, dy = x[1]-x[0], y[1]-y[0]
X, Y = np.meshgrid(x, y, indexing='ij')

L = FinDiff(0, dx, 2) + FinDiff(1, dy, 2)
f = np.zeros(shape)

bc = BoundaryConditions(shape)
bc[1,:] = FinDiff(0, dx, 1), 0  # Neumann BC
bc[-1,:] = 300. - 200*Y   # Dirichlet BC
bc[:, 0] = 300.   # Dirichlet BC
bc[1:-1, -1] = FinDiff(1, dy, 1), 0  # Neumann BC

pde = PDE(L, f, bc)
u = pde.solve()
```

Result:

<p align="center">
<img src="docs/frontpage/heat.png"/>
</p>

## Citations

You have used *findiff* in a publication? Here is how you can cite it:

> M. Baer. *findiff* software package. URL: https://github.com/maroba/findiff. 2018

BibTeX entry:

```
@misc{findiff,
  title = {{findiff} Software Package},
  author = {M. Baer},
  url = {https://github.com/maroba/findiff},
  key = {findiff},
  note = {\url{https://github.com/maroba/findiff}},
  year = {2018}
}
```

## Development

### Set up development environment

- Fork the repository
- Clone your fork to your machine
- Install in development mode:

```
pip install -e .
```

### Running tests

From the console:

```
pip install pytest
pytest tests
```
