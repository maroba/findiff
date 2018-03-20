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
    elif ndims == 2:
        if order == 1:
            return _diff_2d_ord1(y, h, acc, dims)
        elif order == 2:
            return _diff_2d_ord2(y, h, acc, dims)
        else:
            raise AttributeError("This order is not yet implemented")
    elif ndims == 3:
        if order == 1:
            return _diff_3d_ord1(y, h, acc, dims)
        elif order == 2:
            return _diff_3d_ord2(y, h, acc, dims)
        else:
            raise AttributeError("This order is not yet implemented")
    else:
        raise AttributeError("Dimensions > 2 not yet implemented")


def _diff_general(y, h, deriv, dim, acc):

    coefs = coefficients(deriv, acc)

    # 1D
    scheme = "center"
    weights = coefs[scheme]["coefficients"]
    offsets = coefs[scheme]["offsets"]

    npts = y.shape[dim]

    nbndry = len(weights) // 2
    ref_slice = slice(nbndry, npts - nbndry, 1)
    off_slices = [ _shift_slice(ref_slice, offsets[k], npts) for k in range(len(offsets))]

    yd = np.zeros_like(y)

    for w, s in zip(weights, off_slices):
        yd[ref_slice] += w * y[s]

    scheme = "forward"
    weights = coefs[scheme]["coefficients"]
    offsets = coefs[scheme]["offsets"]

    ref_slice = slice(0, nbndry, 1)
    off_slices = [_shift_slice(ref_slice, offsets[k], npts) for k in range(len(offsets))]

    for w, s in zip(weights, off_slices):
        yd[ref_slice] += w * y[s]

    scheme = "backward"
    weights = coefs[scheme]["coefficients"]
    offsets = coefs[scheme]["offsets"]

    ref_slice = slice(npts - nbndry, npts, 1)
    off_slices = [_shift_slice(ref_slice, offsets[k], npts) for k in range(len(offsets))]

    for w, s in zip(weights, off_slices):
        yd[ref_slice] += w * y[s]

    h_inv = 1./h**deriv
    return yd * h_inv


def _shift_slice(sl, off, max_index):

    if sl.start + off < 0 or sl.stop + off > max_index:
        raise IndexError("Shift slice out of bounds")

    return slice(sl.start + off, sl.stop + off, sl.step)


def _diff_2d_ord1(y, h, acc, dims):

    yd = np.zeros_like(y)

    h_inv = 1. / h

    if acc == 2:
        if dims == 0:
            yd[1:-1, :] = 0.5 * (-y[0:-2, :] + y[2:, :])
            yd[0:1, :] = -1.5 * y[0:1, :] + 2. * y[1:2, :] - 0.5 * y[2:3, :]
            yd[-1:, :] = 1.5 * y[-1:, :] - 2. * y[-2:-1, :] + 0.5 * y[-3:-2, :]
        elif dims == 1:
            yd[:, 1:-1] = 0.5  * (-y[:, 0:-2] + y[:, 2:])
            yd[:, 0:1] = -1.5 * y[:, 0:1] + 2. * y[:, 1:2] - 0.5 * y[:, 2:3]
            yd[:, -1:] = 1.5 * y[:, -1:] - 2. * y[:, -2:-1] + 0.5 * y[:, -3:-2]
        else:
            raise ValueError("No such dimension")
    else:
        raise AttributeError("Only accuracy order 2 implemented")

    return yd * h_inv[dims]


def _diff_2d_ord2(y, h, acc, dims):

    yd = np.zeros_like(y)

    h2_inv = 1. / h**2

    if acc == 2:

        if (dims == 0).all():
            yd[1:-1, :] = y[0:-2, :] - 2. * y[1:-1, :] + y[2:, :]
            yd[0:1, :] = 2. * y[0:1, :] - 5 * y[1:2, :] + 4. * y[2:3, :] - y[3:4, :]
            yd[-1:, :] = 2. * y[-1:, :] - 5 * y[-2:-1, :] + 4 * y[-3:-2, :] - y[-4:-3, :]

        elif (dims == 1).all():
            yd[:, 1:-1] = y[:, 0:-2] - 2. * y[:, 1:-1] + y[:, 2:]
            yd[:, 0:1] = 2. * y[:, 0:1] - 5 * y[:, 1:2] + 4. * y[:, 2:3] - y[:, 3:4]
            yd[:, -1:] = 2. * y[:, -1:] - 5 * y[:, -2:-1] + 4 * y[:, -3:-2] - y[:, -4:-3]
        else:
            raise AttributeError("This combination of derivatives is not implemented")

    else:
        raise AttributeError("Only accuracy order 2 implemented")

    return yd * h2_inv[dims[0]]


def _diff_3d_ord1(y, h, acc, dims):

    yd = np.zeros_like(y)

    h_inv = 1. / h

    if acc == 2:
        if dims == 0:
            yd[1:-1, :, :] = 0.5 * (-y[0:-2, :, :] + y[2:, :, :])
            yd[0:1, :, :] = -1.5 * y[0:1, :, :] + 2. * y[1:2, :, :] - 0.5 * y[2:3, :, :]
            yd[-1:, :, :] = 1.5 * y[-1:, :, :] - 2. * y[-2:-1, :, :] + 0.5 * y[-3:-2, :, :]
        elif dims == 1:
            yd[:, 1:-1, :] = 0.5  * (-y[:, 0:-2, :] + y[:, 2:, :])
            yd[:, 0:1, :] = -1.5 * y[:, 0:1, :] + 2. * y[:, 1:2, :] - 0.5 * y[:, 2:3, :]
            yd[:, -1:, :] = 1.5 * y[:, -1:, :] - 2. * y[:, -2:-1, :] + 0.5 * y[:, -3:-2, :]
        elif dims == 2:
            yd[:, :, 1:-1] = 0.5  * (-y[:, :, 0:-2] + y[:, :, 2:])
            yd[:, :, 0:1] = -1.5 * y[:, :, 0:1] + 2. * y[:, :, 1:2] - 0.5 * y[:, :, 2:3]
            yd[:, :, -1:] = 1.5 * y[:, :, -1:] - 2. * y[:, :, -2:-1] + 0.5 * y[:, :, -3:-2]
        else:
            raise ValueError("No such dimension")
    else:
        raise AttributeError("Only accuracy order 2 implemented")

    return yd * h_inv[dims]


def _diff_3d_ord2(y, h, acc, dims):

    yd = np.zeros_like(y)

    h2_inv = 1. / h**2

    if acc == 2:

        if (dims == 0).all():
            yd[1:-1, :, :] = y[0:-2, :, :] - 2. * y[1:-1, :, :] + y[2:, :, :]
            yd[0:1, :, :] = 2. * y[0:1, :, :] - 5 * y[1:2, :, :] + 4. * y[2:3, :, :] - y[3:4, :, :]
            yd[-1:, :, :] = 2. * y[-1:, :, :] - 5 * y[-2:-1, :, :] + 4 * y[-3:-2, :, :] - y[-4:-3, :, :]
        elif (dims == 1).all():
            yd[:, 1:-1, :] = y[:, 0:-2, :] - 2. * y[:, 1:-1, :] + y[:, 2:, :]
            yd[:, 0:1, :] = 2. * y[:, 0:1, :] - 5 * y[:, 1:2, :] + 4. * y[:, 2:3, :] - y[:, 3:4, :]
            yd[:, -1:, :] = 2. * y[:, -1:, :] - 5 * y[:, -2:-1, :] + 4 * y[:, -3:-2, :] - y[:, -4:-3, :]
        elif (dims == 2).all():
            yd[:, :, 1:-1] = y[:, :, 0:-2] - 2. * y[:, :, 1:-1] + y[:, :, 2:]
            yd[:, :, 0:1] = 2. * y[:, :, 0:1] - 5 * y[:, :, 1:2] + 4. * y[:, :, 2:3] - y[:, :, 3:4]
            yd[:, :, -1:] = 2. * y[:, :, -1:] - 5 * y[:, :, -2:-1] + 4 * y[:, :, -3:-2] - y[:, :, -4:-3]
        else:
            raise AttributeError("This combination of derivatives is not implemented")

    else:
        raise AttributeError("Only accuracy order 2 implemented")

    return yd * h2_inv[dims[0]]
