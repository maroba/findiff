# findiff
A Python package for finite difference numerical derivatives in 1D, 2D and 3D.

## Usage

### Derivatives in 1D

```python
import numpy as np
import findiff

# Our 1D grid:
nx = 100
x = np.linspace(-1, 1, nx)
dx = x[1] - x[0]

# Init the array to differentiate
f = init_your_1d_array_to_differentiate()

# First derivative
df_dx = findiff.diff(f, dx, dims=[0])

# Second derivative
d2f_dx2 = findiff.diff(f, dx, dims=[0, 0])

# 30th derivative
d30f_dx30 = findiff.diff(f, dx, dims=[0]*30)

```

### Derivatives in 2D

```python
import numpy as np
import findiff

# Our 2D grid:
nx = 100
ny = 100
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
dx = x[1] - x[0]
dy = y[1] - y[0] 

# Init the array to differentiate
f = init_your_2d_array_to_differentiate()

# First derivative with respect to first coordinate
df_dx = findiff.diff(f, h=[dx, dy], dims=[0])

# Second derivative with respect to the first coordinate
d2f_dx2 = findiff.diff(f, h=[dx, dy], dims=[0, 0])

# Second derivative with respect to the first, then the second coordinate
d2f_dxdy = findiff.diff(f, h=[dx, dy], dims=[0, 1])
```

### Derivatives in 3D

```python
import numpy as np
import findiff

# Our 3D grid:
nx = 100
ny = 100
nz = 100
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
z = np.linspace(-1, 1, nz)
dx = x[1] - x[0]
dy = y[1] - y[0] 
dz = z[1] - z[0] 

# Init the array to differentiate
f = init_your_3d_array_to_differentiate()

# First derivatives
df_dx = findiff.diff(f, h=[dx, dy, dz], dims=[0])
df_dy = findiff.diff(f, h=[dx, dy, dz], dims=[1])
df_dz = findiff.diff(f, h=[dx, dy, dz], dims=[2])

# Second derivative with respect to the third coordinate
d2f_dx2 = findiff.diff(f, h=[dx, dy, dz], dims=[2, 2])

# 30th derivative with respect to the second coordinate
d2f_dx2 = findiff.diff(f, h=[dx, dy, dz], dims=[1]*30)


```

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

