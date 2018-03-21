# findiff
A Python package for finite difference numerical derivatives in
any number of dimensions.

## Quickstart

### Derivatives

#### Derivatives in 1D

```python
import findiff

# Init the array to differentiate
f = init_your_1d_array_to_differentiate()
g = init_some_other_1d_array_to_differentiate()

# First derivative, dx is the grid spacing along dimension 0
d_dx = findiff.FinDiff(h=dx, dims=[0])
df_dx = d_dx(f)
dg_dx = d_dx(g)

# Second derivative
d2_dx2 = findiff.FinDiff(dx, dims=[0, 0])
d2f_dx2 = d2_dx2(f)

# 6th derivative
d6_dx6 = findiff.FinDiff(dx, dims=[0]*6)
d6f_dx6 = d6_dx6(f)
```

#### Derivatives in 2D

```python
import findiff

# Init the array to differentiate
f = init_your_2d_array_to_differentiate()

# The first derivatives.
# dx, dy are the grid spacings along dimensions 0 and 1
d_dx = findiff.FinDiff(h=[dx, dy], dims=[0])
df_dx = d_dx(f)

d_dy = findiff.FinDiff(h=[dx, dy], dims=[1])
df_dy = d_dy(f)


# Second derivatives
d2_dx2 = findiff.FinDiff(h=[dx, dy], dims=[0, 0])
d2f_dx2 = d2_dx2(f)

d2_dy2 = findiff.FinDiff(h=[dx, dy], dims=[1, 1])
d2f_dy2 = d2_dy2(f)

d2_dxdy = findiff.FinDiff(h=[dx, dy], dims=[0, 1])
d2f_dxdy = d2_dxdy(f)
```

#### Derivatives in 3D

```python
import findiff

# Init the array to differentiate
f = init_your_3d_array_to_differentiate()

# First derivatives
d_dx = findiff.FinDiff(h=[dx, dy, dz], dims=[0])
df_dx = d_dx(f)

d_dy = findiff.FinDiff(h=[dx, dy, dz], dims=[1])
df_dy = d_dy(f)

d_dz = findiff.FinDiff(h=[dx, dy, dz], dims=[2])
df_dz = d_dz(f)

# Second derivatives
d2_dx2 = findiff.FinDiff(h=[dx, dy, dz], dims=[0, 0])
d2f_dx2 = d2_dx2(f)

d2_dy2 = findiff.FinDiff(h=[dx, dy, dz], dims=[1, 1])
d2f_dy2 = d2_dy2(f)

d2_dz2 = findiff.FinDiff(h=[dx, dy, dz], dims=[2, 2])
d2f_dz2 = d2_dz2(f)

# 8th derivative with respect to the second coordinate
d8_dy8 = findiff.FinDiff(h=[dx, dy, dz], dims=[1]*8)
d8f_dy8 = d8_dy8(f)

```

#### Derivatives in N dimensions

The package can work with any number of dimensions, the generalization
of usage is straight forward. The only limit is memory and CPU speed.


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



## Dependencies

The only dependency used is numpy. You can install numpy using pip:

```
pip install numpy
```

