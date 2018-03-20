import numpy as np


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
        if order == 1:
            return _diff_1d_ord1(y, h, acc)
        elif order == 2:
            return _diff_1d_ord2(y, h, acc)
        else:
            raise AttributeError("This order is not yet implemented")
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


def _diff_1d_ord1(y, h, acc):

    yd = np.zeros_like(y)

    if acc == 2:
        yd[1:-1] = 0.5 * (-y[0:-2] + y[2:])
        yd[0:1] = -1.5 * y[0:1] + 2. * y[1:2] - 0.5 * y[2:3]
        yd[-1:] = 1.5 * y[-1:] - 2. * y[-2:-1] + 0.5 * y[-3:-2]

    elif acc == 4:

        coefs = [1./12, -2./3, 2./3, -1./12]
        slices = [slice(0, -4, 1), slice(1, -3, 1), slice(3, -1, 1), slice(4, None, 1)]

        for c, s in zip(coefs, slices):
            yd[2:-2] += c * y[s]

        coefs = [-25./12, 4, -3., 4./3, -1./4]
        slices = [slice(k, k+2, 1) for k in range(5)]

        for c, s in zip(coefs, slices):
            yd[0:2] += c * y[s]

        coefs = reversed([25./12, -4, 3., -4./3, 1./4])
        slices = reversed([slice(-2, None, 1), slice(-3, -1, 1), slice(-4, -2, 1), slice(-5, -3, 1), slice(-6, -4, 1)])

        for c, s in zip(coefs, slices):
            yd[-2:] += c * y[s]

    else:
        raise AttributeError("Only accuracy orders 2, 4 implemented for first derivatives")

    h_inv = 1./h
    return yd * h_inv


def _diff_1d_ord2(y, h, acc):

    yd = np.zeros_like(y)

    if acc == 2:

        yd[1:-1] = y[0:-2] - 2.*y[1:-1] + y[2:]
        yd[0:1] = 2.*y[0:1] - 5*y[1:2] + 4.*y[2:3] - y[3:4]
        yd[-1:] = 2.*y[-1:] - 5*y[-2:-1] + 4*y[-3:-2] - y[-4:-3]

    elif acc == 4:

        coefs = [-1./12, 4./3, -2.5, 4./3, -1./12]
        slices = [slice(0, -4, 1), slice(1, -3, 1), slice(2, -2, 1), slice(3, -1, 1), slice(4, None, 1)]

        for c, s in zip(coefs, slices):
            yd[2:-2] += c * y[s]

        coefs = ([15./4, -77./6, 107./6, -13., 61./12, -5./6])
        slices = [slice(k, k+2, 1) for k in range(6)]

        for c, s in zip(coefs, slices):
            yd[0:2] += c * y[s]

        coefs = reversed([15./4, -77./6, 107./6, -13., 61./12, -5./6])
        slices = reversed([slice(-2, None, 1), slice(-3, -1, 1), slice(-4, -2, 1),
                  slice(-5, -3, 1), slice(-6, -4, 1), slice(-7, -5, 1)])

        for c, s in zip(coefs, slices):
            yd[-2:] += c * y[s]

    else:
        raise AttributeError("Only accuracy orders 2, 4 implemented for second derivatives")

    h2_inv = 1./h**2
    return yd * h2_inv


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
