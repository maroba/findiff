# findiff
[![PyPI version](https://badge.fury.io/py/findiff.svg)](https://badge.fury.io/py/findiff)

A Python package for finite difference numerical derivatives in
any number of dimensions. 

## Features ##

* Differentiate arrays of any number of dimensions along any axis
* Partial derivatives of any desired order
* Can handle uniform and non-uniform grids
* Can handle arbitrary linear combinations of derivatives with constant and variable coefficients
* Accuracy order can be specified
* Calculate raw finite difference coefficients for any order and accuracy for uniform and non-uniform grids

## Installation

Simply use pip:

```
pip install findiff
```

## Quickstart

### Derivatives on uniform grids

_findiff_ works in any number of dimensions. But for the sake of demonstration, suppose you
want to differentiate four-dimensional function given as a 4D array.

```python
from findiff import *

# Init the array to differentiate
f = init_your_4d_array_to_differentiate()

# First derivatives
d_dx = FinDiff(h=[dx, dy, dz, du], dims=[0])
df_dx = d_dx(f)

d_dy = FinDiff(h=[dx, dy, dz, du], dims=[1])
df_dy = d_dy(f)

d_dz = FinDiff(h=[dx, dy, dz, du], dims=[2])
df_dz = d_dz(f)

d_du = FinDiff(h=[dx, dy, dz, du], dims=[3])
df_du = d_du(f)

# Second derivatives
d2_dx2 = FinDiff(h=[dx, dy, dz, du], dims=[0, 0])
d2f_dx2 = d2_dx2(f)

d2_dz2 = FinDiff(h=[dx, dy, dz, du], dims=[2, 2])
d2f_dz2 = d2_dz2(f)

# 8th derivative with respect to the second coordinate
d8_dy8 = FinDiff(h=[dx, dy, dz, du], dims=[1]*8)
d8f_dy8 = d8_dy8(f)

# Mixed 3rd derivatives, twice with respect to x, once w.r.t. z
d3_dx2dz = FinDiff(h=[dx, dy, dz, du], dims=[0, 0, 2])

# You can also create linear combinations of differential operators
h = [dx, dy, dz, du]
diff_op = Coefficient(2) * FinDiff(h=h, dims=[0, 0, 2] + Coefficient(3) * FinDiff(h=h, dims=[1, 1, 0])

```

More examples, including linear combinations with variable coefficients can be found [here](examples).


#### Derivatives in N dimensions

The package can work with any number of dimensions, the generalization
of usage is straight forward. The only limit is memory and CPU speed.

### Derivatives on non-uniform grids

_findiff_ can also handle non-uniform grids. The only difference is that instead of giving 
the grid spacing to the `FinDiff` constructor, you give it the coordinates:

```python
import numpy as np
from findiff import FinDiff

# A non-uniform 3D grid:
x = np.r_[np.arange(0, 4, 0.05), np.arange(4, 10, 1)]
y = np.r_[np.arange(0, 4, 0.05), np.arange(4, 10, 1)]
z = np.r_[np.arange(0, 4.5, 0.05), np.arange(4.5, 10, 1)]
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Some function to differentiate
f = np.exp(-X**2-Y**2-Z**2)

# Define the partial derivative with respect to y, e.g.
d_dy = FinDiff(coords=[x, y, z], dims=[1])

# Apply it to f
fy = d_dy(f)
```

### Accuracy Control

When constructing an instance of `FinDiff`, you can request the desired accuracy
order by setting the parameter `acc`. 

```
d2_dx2 = findiff.FinDiff(h=[dx, dy], dims=[1, 1], acc=4)
d2f_dx2 = d2_dx2(f)
```

If not specified, second order accuracy will be taken by default.


### Finite Difference Coefficients

Sometimes you may want to have the raw finite difference coefficients.
These can be obtained for __any__ derivative and accuracy order
using `findiff.coefficients(deriv, acc)`. For instance,

```python
import findiff
coefs = findiff.coefficients(deriv=2, acc=2)
```

gives

```
{ 'backward': {'coefficients': array([-1.,  4., -5.,  2.]),
               'offsets': array([-3, -2, -1,  0])},
  'center': {'coefficients': array([ 1., -2.,  1.]),
             'offsets': array([-1,  0,  1])},
  'forward': {'coefficients': array([ 2., -5.,  4., -1.]),
              'offsets': array([0, 1, 2, 3])}
              }
```

## Further examples

Here is a collection of further examples using the _findiff_ package:

* [Basic usage](examples/basic.py)
* [Linear operators](examples/linear_op.py)
* [Non-uniform grids](examples/non-uniform-grids.ipynb)
