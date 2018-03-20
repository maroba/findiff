# findiff
A Python package for finite difference numerical derivatives in 1D, 2D and 3D.

## Usage

### Derivatives in 1D

```python
import numpy as np
import findiff

# Our 1D grid:
nx = 100
x = np.linspace(-1, 1, 100)
dx = x[1] - x[0]

# Init the array to differentiate
y = init_your_1d_array_to_differentiate()

# First derivative with accuracy order 2
dy_dx = findiff.diff(y, dx, order=1, acc=2)

# Second derivative with accuracy order 2
d2y_dx2 = findiff.diff(y, dx, order=2, acc=2)

# Second derivative with accuracy order 4
d2y_dx2 = findiff.diff(y, dx, order=2, acc=4)
```

## Dependencies

The only dependency used is numpy. You can install numpy using pip:

```
pip install numpy
```

