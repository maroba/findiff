import numpy as np
from findiff.coefs import coefficients, coefficients_non_uni


class FinDiff(object):
    """Wrapper class for finite difference linear differential operators in any number of dimensions
       on uniform and non-uniform grids."""

    def __init__(self, dims=[0], **kwargs):
        """Constructor for FinDiff class
        
            Parameters:
            -----------
            
            dims        list / array-like
                        A list with the dimensions along which to take derivatives. Repetition means
                        higher derivative. For instance, dims=[0] means first derivative along 0-th
                        axis, dims=[0,0,1] means third partial derivative, twice along 0-th axis, once
                        along 1st axis
            
            **kwargs:
            
                h       list / array-like
                        A list of real numbers with the grid spacings along all the axes in case
                        of uniform grids. The length of the list signifies the dimension, i.e. number
                        of independent variables. 
                
                coords  list / array-like
                        A list of 1D-arrays of real numbers with the coordinate values along each axis.
                        This signifies that you are using a non-uniform grid.
                        
                acc     even integer
                        The desired accuracy order. Default is acc=2."""

        if "h" in kwargs:   # we have a uniform grid
            if "coords" in kwargs:
                raise Exception("Either specify h or coords, not both.")
            h = kwargs["h"]
            self.uniform = True
        elif "coords" in kwargs:  # we have a non-uniform grid
            if "h" in kwargs:
                raise Exception("Either specify h or coords, not both.")
            coords = kwargs["coords"]
            self.uniform = False
        else:
            if "empty" not in kwargs:
                raise Exception("Neither h nor coords specified.")


        if "acc" in kwargs:
            acc = kwargs["acc"]
        else:
            acc = 2

        if "empty" in kwargs and kwargs["empty"]:
            self._basic_ops = []
            self._coefs = []
        else:
            if self.uniform:
                self._basic_ops = [BasicFinDiff(h, dims, acc)]
            else:
                self._basic_ops = [BasicFinDiffNonUniform(coords, dims, acc)]

            self._coefs = [Coefficient(1)]

    def __call__(self, y):
        """Applies the linear differential operator to y
        
            Parameters:
            -----------
            
                y       ndarray
                        The array to differentiate
                        
            Returns:
            --------
            
                An ndarray with the derivative. It has the same shape as y. """

        result = np.zeros_like(y)

        for c, op in zip(self._coefs, self._basic_ops):

            if isinstance(c.value, np.ndarray) or c.value != 1:
                result += c.value * op(y)
            else:
                result += op(y)

        return result

    def __add__(self, other):

        if self._grids_are_incompatible(other):
            raise ValueError("Operators on incompatible grids cannot be added.")

        new_op = FinDiff(empty=True)
        new_op._basic_ops.extend(self._basic_ops)
        new_op._coefs.extend(self._coefs)
        new_op._basic_ops.extend(other._basic_ops)
        new_op._coefs.extend(other._coefs)
        return new_op

    def __mul__(self, other):

        if not isinstance(other, Coefficient):
            other = Coefficient(other)

        return other * self

    def __rmul__(self, other):

        if not isinstance(other, Coefficient):
            other = Coefficient(other)

        new_op = FinDiff(empty=True)
        new_op._basic_ops.extend(self._basic_ops)
        new_op._coefs.extend(self._coefs)

        for i in range(len(new_op._coefs)):
            new_op._coefs[i].value *= other.value

        return new_op

    def _grids_are_incompatible(self, other):
        if self.uniform and other.uniform:
            return (self._basic_ops[0]._h != other._basic_ops[0]._h).any()
        if not self.uniform and not other.uniform:
            coords1 = self._basic_ops[0]._coords
            coords2 = other._basic_ops[0]._coords
            return (coords1 != coords2).any()
        return True


class BasicFinDiff(object):
    """Finite difference derivative of any order, any accuracy in any dimension for uniform grids 
    """

    def __init__(self, h=[1.], dims=[0], acc=2):
        """Constructor for Finite Difference operator on _uniform_ grids
           
           Parameters:
           ----------
           
           h        array-like
                    The grid spacing along each axis.
           dims     array-like
                    The axes along which to take the derivatives. Multiple values mean higher derivative.
           acc      int
                    The accuracy order of the finite difference scheme.
                            
           *Note*: You can use this class, but it is usually better to use the wrapper class FinDiff 
                            
           Example:
           --------
           
           Suppose f is a four-dimensional array. The second partial derivative with respect to the second axis,
                    
                    \frac{\partial^2 f}{\partial y^2},
                     
           on a grid with equidistant spacing h=[0.1, 0.1, 0.1, 0.1] is given by
           
                FinDiff(h=[0.1, 0.1, 0.1, 0.1], dims=[1, 1])
                
        """

        self._h = _wrap_in_array(h)
        self._dims = _wrap_in_array(dims)
        self._acc = acc

        ndims = len(self._h)
        self._derivs = [np.sum(self._dims == i) for i in range(ndims)]
        self._coefs = self._det_coefs()

    def _det_coefs(self):
        """Calculates the finite difference coefficients for the requested partial derivatives"""

        coefs = []

        for i in range(len(self._h)):
            coefs.append(coefficients(self._derivs[i], self._acc))

        return coefs

    def __call__(self, y):
        """Applies the finite difference operator to a function y"""

        ndims = len(y.shape)
        if ndims != len(self._h):
            raise IndexError("y and h have different dimensions")

        yd = np.array(y)

        for i in range(ndims):
            if self._derivs[i] > 0:
                yd = _diff_general(yd, self._h, self._derivs[i], i, self._acc, self._coefs[i])

        return yd


class BasicFinDiffNonUniform(object):
    """Finite difference derivative of any order, any accuracy in any dimension for uniform grids 
    """

    def __init__(self, coords, dims=[0], acc=2):

        self._coords = np.array(coords)
        self._dims = _wrap_in_array(dims)
        self._acc = acc

        ndims = len(self._coords)
        self._derivs = [np.sum(self._dims == i) for i in range(ndims)]
        self._coefs = self._det_coefs()

    def _det_coefs(self):
        """Calculates the finite difference coefficients for the requested partial derivatives"""

        coefs = []

        ndims = len(self._coords)

        for idim in range(ndims):
            c = []
            for i in range(len(self._coords[idim])):
                c.append(coefficients_non_uni(self._derivs[idim], self._acc, self._coords[idim], i))
            coefs.append(c)

        return coefs

    def __call__(self, y):

        ndims = len(y.shape)
        if ndims != len(self._coords):
            raise IndexError("y and h have different dimensions")

        yd = np.array(y)

        for idim in range(ndims):
            if self._derivs[idim] > 0:
                yd = _diff_general_non_uni(yd, self._coords, idim, self._coefs[idim])

        return yd


class Coefficient(object):
    """Encapsulates a constant (number) or variable (coordinate array) value to multiply with a linear operator
    """

    def __init__(self, value):
        self.value = value


class Laplacian(object):
    """A representation of the Laplace operator in arbitrary dimensions using finite difference schemes"""

    def __init__(self, h=[1.], acc=2):
        """Constructor for the Laplacian
        
           Parameters:
           -----------
           
           h        array-like
                    The grid spacing along each axis
           acc      int
                    The accuracy order of the finite difference scheme        
        """

        h = _wrap_in_array(h)

        self._parts = [FinDiff(h=h, dims=[k, k], acc=acc) for k in range(len(h))]

    def __call__(self, f):
        """Applies the Laplacian to the array f
        
           Parameters:
           -----------
           
           f        ndarray
                    The function to differentiate given as an array.
        
           Returns:
           --------    
           
           an ndarray with Laplace(f)
        
        """
        laplace_f = np.zeros_like(f)

        for part in self._parts:
            laplace_f += part(f)

        return laplace_f


def _diff_general_non_uni(y, coords, dim, coefs):
    """The core function to take a partial derivative on a non-uniform grid"""

    yd = np.zeros_like(y)

    ndims = len(y.shape)
    multi_slice = [slice(None, None)] * ndims
    ref_multi_slice = [slice(None, None)] * ndims

    for i, x in enumerate(coords[dim]):
        weights = coefs[i]["coefficients"]
        offsets = coefs[i]["offsets"]
        ref_multi_slice[dim] = i

        for off, w in zip(offsets, weights):
            multi_slice[dim] = i + off
            yd[ref_multi_slice] += w * y[multi_slice]

    return yd


def _diff_general(y, h, deriv, dim, acc, coefs=None):
    """The core function to take a partial derivative on a uniform grid.
    """

    if coefs is None:
        coefs = coefficients(deriv, acc)

    npts = y.shape[dim]

    scheme = "center"
    weights = coefs[scheme]["coefficients"]
    offsets = coefs[scheme]["offsets"]

    nbndry = len(weights) // 2
    ref_slice = slice(nbndry, npts - nbndry, 1)
    off_slices = [_shift_slice(ref_slice, offsets[k], npts) for k in range(len(offsets))]

    yd = np.zeros_like(y)

    _apply_to_array(yd, y, weights, off_slices, ref_slice, dim)

    scheme = "forward"
    weights = coefs[scheme]["coefficients"]
    offsets = coefs[scheme]["offsets"]

    ref_slice = slice(0, nbndry, 1)
    off_slices = [_shift_slice(ref_slice, offsets[k], npts) for k in range(len(offsets))]

    _apply_to_array(yd, y, weights, off_slices, ref_slice, dim)

    scheme = "backward"
    weights = coefs[scheme]["coefficients"]
    offsets = coefs[scheme]["offsets"]

    ref_slice = slice(npts - nbndry, npts, 1)
    off_slices = [_shift_slice(ref_slice, offsets[k], npts) for k in range(len(offsets))]

    _apply_to_array(yd, y, weights, off_slices, ref_slice, dim)

    h_inv = 1./h**deriv
    return yd * h_inv[dim]


def _apply_to_array(yd, y, weights, off_slices, ref_slice, dim):
    """Applies the finite differences only to slices along a given axis"""

    ndims = len(y.shape)

    all = slice(None, None, 1)

    ref_multi_slice = [all] * ndims
    ref_multi_slice[dim] = ref_slice

    for w, s in zip(weights, off_slices):
        off_multi_slice = [all] * ndims
        off_multi_slice[dim] = s
        yd[ref_multi_slice] += w * y[off_multi_slice]


def _shift_slice(sl, off, max_index):

    if sl.start + off < 0 or sl.stop + off > max_index:
        raise IndexError("Shift slice out of bounds")

    return slice(sl.start + off, sl.stop + off, sl.step)


def _wrap_in_array(val):

    if hasattr(val, "__len__"):
        return np.array(val)

    return np.array([val])
