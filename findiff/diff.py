import numpy as np
from findiff.coefs import coefficients


class FinDiff(object):
    """Finite difference representation of a derivative of any order, any accuracy in any dimension"""

    def __init__(self, h=[1.], dims=[0], acc=2):
        """Constructor for Finite Difference operator
           
           Parameters:
           ----------
           
           h        array-like
                    The grid spacing along each axis.
           dims     array-like
                    The axes along which to take the derivatives. Multiple values mean higher derivative.
           acc      int
                    The accuracy order of the finite difference scheme.
                            
                            
           Example:
           --------
           
           Suppose f is a four-dimensional array. The second partial derivative with respect to the second axis,
                    
                    \frac{\partial^2 f}{\partial z^2},
                     
           on a grid with equidistant spacing h=[0.1, 0.1, 0.1, 0.1] is given by
           
                FinDiff(h=[0.1, 0.1, 0.1, 0.1], dims=[1, 1])
                
        """

        if not hasattr(h, "__len__"):
            self._h = np.array([h])
        else:
            self._h = np.array(h)

        if not hasattr(dims, "__len__"):
            self._dims = np.array([dims])
        else:
            self._dims = np.array(dims)

        self._acc = acc
        ndims = len(self._h)
        self._derivs = [np.sum(self._dims == i) for i in range(ndims)]
        self._coefs = []
        for i in range(ndims):
            self._coefs.append(coefficients(self._derivs[i], acc))

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

        if not hasattr(h, "__len__"):
            h = np.array([h])
        else:
            h = np.array(h)

        self._parts = [FinDiff(h, dims=[k, k], acc=acc) for k in range(len(h))]

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
            laplace_f += part.diff(f)

        return laplace_f


def _diff_general(y, h, deriv, dim, acc, coefs=None):

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

