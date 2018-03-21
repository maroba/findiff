# findiff
A Python package for finite difference numerical derivatives in
any number of dimensions.

## Quickstart


### findiff.diff

#### Derivatives in 1D

```python
import findiff

# Init the array to differentiate
f = init_your_1d_array_to_differentiate()

# First derivative, dx is the grid spacing along dimension 0
df_dx = findiff.diff(f, dx, dims=[0])

# Second derivative
d2f_dx2 = findiff.diff(f, dx, dims=[0, 0])

# 6th derivative
d6f_dx6 = findiff.diff(f, dx, dims=[0]*6)
```

#### Derivatives in 2D

```python
import findiff

# Init the array to differentiate
f = init_your_2d_array_to_differentiate()

# The first derivatives.
# dx, dy are the grid spacings along dimensions 0 and 1
df_dx = findiff.diff(f, h=[dx, dy], dims=[0])
df_dy = findiff.diff(f, h=[dx, dy], dims=[1])

# Second derivatives
d2f_dx2 = findiff.diff(f, h=[dx, dy], dims=[0, 0])
d2f_dy2 = findiff.diff(f, h=[dx, dy], dims=[1, 1])
d2f_dxdy = findiff.diff(f, h=[dx, dy], dims=[0, 1])

```

#### Derivatives in 3D

```python
import findiff

# Init the array to differentiate
f = init_your_3d_array_to_differentiate()

# First derivatives
df_dx = findiff.diff(f, h=[dx, dy, dz], dims=[0])
df_dy = findiff.diff(f, h=[dx, dy, dz], dims=[1])
df_dz = findiff.diff(f, h=[dx, dy, dz], dims=[2])

# Second derivatives
d2f_dx2 = findiff.diff(f, h=[dx, dy, dz], dims=[0, 0])
d2f_dy2 = findiff.diff(f, h=[dx, dy, dz], dims=[1, 1])
d2f_dz2 = findiff.diff(f, h=[dx, dy, dz], dims=[2, 2])

# 8th derivative with respect to the second coordinate
d8f_dy8 = findiff.diff(f, h=[dx, dy, dz], dims=[1]*8)

```

#### Derivatives in N dimensions

The package can work with any number of dimensions, the generalization
of usage is straight forward. The only limit is memory and CPU speed.


### Accuracy Control

In every application of `diff`, you can request the desired accuracy
order by setting the parameter `acc`. 

```
d2f_dx2 = findiff.diff(f, h=[dx, dy], dims=[1, 1], acc=4)
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

__Note__ that every call to `findiff.diff` with default arguments
computes the finite difference coefficients again. To avoid that, use
the class `FinDiff` instead. It retains the state, so that the coefficients
do not have to be computed again and again.

### Class FinDiff

When you want to apply the same derivative multiple times, it is more
economic to use the `FinDiff` class instead of the `diff` function:

```python
import findiff

f = init_your_ndim_array_to_differentiate()
g = init_some_other_ndim_array_to_differentiate()

spacing = [h0, h1, h2, ...]

d_dy = findiff.FinDiff(h=spacing, dims=[1])

df_dy = d_dy.diff(f)
dg_dy = d_dy.diff(g)

```


## Dependencies

The only dependency used is numpy. You can install numpy using pip:

```
pip install numpy
```

