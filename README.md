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

# 30th derivative
d30f_dx30 = findiff.diff(f, dx, dims=[0]*30)
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

# 30th derivative with respect to the second coordinate
d2f_dy2 = findiff.diff(f, h=[dx, dy, dz], dims=[1]*30)

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


## Dependencies

The only dependency used is numpy. You can install numpy using pip:

```
pip install numpy
```

