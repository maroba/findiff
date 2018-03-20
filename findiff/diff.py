import numpy as np
from findiff.coefs import coefficients


def diff(y, h, order=1, acc=2, dims=0):
    """Returns derivative of a sampled function using finite difference schemes 
    
    
       Parameters
       ----------
       y:   numpy ndarray in 1D, 2D or 3D
            The function to differentiate sampled at equidistant points.
       h:   real number or array-like, depending on whether y is 1D or higher
            The grid spacing.
       order:   int
            The order of the derivative.
       acc:     even int
            The accuracy order.
       dims:  int or array-like depending on whether y is 1D or higher   
            For 2D or 3D the dimension along which to differentiate
            
       Returns
       -------
          a numpy array with the derivative
    """

    ndims = len(y.shape)

    if not isinstance(y, np.ndarray):
        y = np.array(y)

    if ndims == 1:
        h = np.array([h])
    else:
        h = np.array(h)

    if len(h) != ndims:
        raise ValueError("Dimensions of y and h do not match")
    h = np.array(h)
    dims = np.array(dims)

    if ndims == 1:
        return _diff_general(y, h, order, 0, acc)
    else:
        return _diff_general(y, h, order, dims[0], acc)


def _diff_general(y, h, deriv, dim, acc):

    coefs = coefficients(deriv, acc)

    scheme = "center"
    weights = coefs[scheme]["coefficients"]
    offsets = coefs[scheme]["offsets"]

    npts = y.shape[dim]

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

    ndims = len(y.shape)

    if ndims == 1:
        for w, s in zip(weights, off_slices):
            yd[ref_slice] += w * y[s]
    elif ndims == 2:
        if dim == 0:
            for w, s in zip(weights, off_slices):
                yd[ref_slice, :] += w * y[s, :]
        elif dim == 1:
            for w, s in zip(weights, off_slices):
                yd[:, ref_slice] += w * y[:, s]
    elif ndims == 3:
        if dim == 0:
            for w, s in zip(weights, off_slices):
                yd[ref_slice, :, :] += w * y[s, :, :]
        elif dim == 1:
            for w, s in zip(weights, off_slices):
                yd[:, ref_slice, :] += w * y[:, s, :]
        elif dim == 2:
            for w, s in zip(weights, off_slices):
                yd[:, :, ref_slice] += w * y[:, :, s]
    else:
        raise Exception("Only 1D, 2D and 3D implemented")


def _shift_slice(sl, off, max_index):

    if sl.start + off < 0 or sl.stop + off > max_index:
        raise IndexError("Shift slice out of bounds")

    return slice(sl.start + off, sl.stop + off, sl.step)

