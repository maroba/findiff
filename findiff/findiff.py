from copy import deepcopy
import numpy as np
from findiff.operators import Plus, Minus, Operator, UnaryOperator, Multiply
from findiff.coefs import coefficients, coefficients_non_uni


class FinDiff(UnaryOperator):

    def __init__(self, *args, **kwargs):
        """
            A representation of a general linear differential operator expressed in finite differences.
            
            FinDiff objects can be added with other FinDiff objects. They can be multiplied by
            objects of type Coefficient.
                        
            FinDiff is callable, i.e. to apply the derivative, just call the object on the array to
            differentiate.       
       
            :param args: variable number of tuples. Defines what derivative to take.         
                If only one tuple is given, you can leave away the tuple parentheses.
         
            Each tuple has the form
            
                   `(axis, spacing, count)`     for uniform grids
                   
                   `(axis, count)`              for non-uniform grids.
                   
                 `axis` is the dimension along which to take derivative.
             
                 `spacing` is the grid spacing of the uniform grid along that axis.
                                    
                 `count` is the order of the derivative, which is optional an defaults to 1. 
                    
        
            :param kwargs:  variable number of keyword arguments
        
                Allowed keywords:
            
                `acc`:    even integer
                      The desired accuracy order. Default is acc=2.
                        

        ============                                    
        **Example**:
    
        
           For this example, we want to operate on some 3D array f:
           
           >>> import numpy as np
           >>> x, y, z = [np.linspace(-1, 1, 100) for _ in range(3)]
           >>> X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
           >>> f = X**2 + Y**2 + Z**2
        
           To create :math:`\frac{\partial f}{\partial x}` on a uniform grid with spacing dx, dy
           along the 0th axis or 1st axis, respectively, instantiate a FinDiff object and call it:
           
           >>> d_dx = FinDiff(0, dx)
           >>> d_dy = FinDiff(1, dx)
           >>> result = d_dx(f)
           
           For :math:`\frac{\partial^2 f}{\partial x^2}` or :math:`\frac{\partial^2 f}{\partial y^2}`:
           
           >>> d2_dx2 = FinDiff(0, dx, 2)
           >>> d2_dy2 = FinDiff(1, dy, 2)
           >>> result_2 = d2_dx2(f)
           >>> result_3 = d2_dy2(f)
           
           For :math:`\frac{\partial^4 f}{\partial x \partial^2 y \partial z}`, do:
           
           >>> op = FinDiff((0, dx), (1, dy, 2), (2, dz))
           >>> result_4 = op(f)
       
        
        """

        self.acc = None

        for kw in kwargs:
            if kw == "acc":
                self.acc = kwargs[kw]
                if self.acc % 2 == 1:
                    self.acc += 1
            else:
                raise Exception("No such keyword argument: %s" % kw)

        self.root = PartialDerivative(*args, **kwargs)

        self.child = None

    def __call__(self, u, **kwargs):
        """Applies the linear differential operator to y

            Parameters:
            -----------

                y       ndarray
                        The array to differentiate

            Returns:
            --------

                An ndarray with the derivative. It has the same shape as y. """

        for kwarg in kwargs:
            if kwarg == "acc":
                self.set_accuracy(kwargs[kwarg])
            else:
                raise Exception("Unknown kwarg.")

        if self.acc is None:
            self.acc = 2

        if self.child is not None:
            u = self.child.apply(self, u)

        return self.root.apply(self, u)

    def set_accuracy(self, acc):
        """ Sets the accuracy order of the finite difference scheme.
            If the FinDiff object is not a raw partial derivative but a composition of derivatives
            the accuracy order will be propagated to the child operators.
        """
        self.acc = acc
        if self.child:
            self.child.set_accuracy(acc)

    def __add__(self, other):
        """Add FinDiff object with other FinDiff object to linear combination.

           Both FinDiff objects must use the same grid.
        """

        fd = deepcopy(self)
        fd.root = Plus(fd.root, deepcopy(other))
        return fd

    def __sub__(self, other):
        """ Subtract one FinDiff object from the other. """

        fd = deepcopy(self)
        fd.root = Minus(fd.root, deepcopy(other))
        return fd

    def __rmul__(self, other):
        """Multiply FinDiff object with object of type Coef or chain FinDiff objects."""

        if isinstance(other, Coef):
            mult = Multiply(other.value, deepcopy(self))
        else:
            mult = Multiply(other, deepcopy(self))
        fd = deepcopy(self)
        fd.root = mult

        return fd

    def __mul__(self, other):
        """Multiply FinDiff object with object of type Coef or chain FinDiff objects."""

        if isinstance(other, Operator):
            self.child = other

        return self

    def apply(self, fd, u):
        return self.root.apply(fd, u)

    def diff(self, y, h, deriv, dim, coefs):
        """The core function to take a partial derivative on a uniform grid.
        """

        npts = y.shape[dim]

        scheme = "center"
        weights = coefs[scheme]["coefficients"]
        offsets = coefs[scheme]["offsets"]

        nbndry = len(weights) // 2
        ref_slice = slice(nbndry, npts - nbndry, 1)
        off_slices = [self._shift_slice(ref_slice, offsets[k], npts) for k in range(len(offsets))]

        yd = np.zeros_like(y)

        self._apply_to_array(yd, y, weights, off_slices, ref_slice, dim)

        scheme = "forward"
        weights = coefs[scheme]["coefficients"]
        offsets = coefs[scheme]["offsets"]

        ref_slice = slice(0, nbndry, 1)
        off_slices = [self._shift_slice(ref_slice, offsets[k], npts) for k in range(len(offsets))]

        self._apply_to_array(yd, y, weights, off_slices, ref_slice, dim)

        scheme = "backward"
        weights = coefs[scheme]["coefficients"]
        offsets = coefs[scheme]["offsets"]

        ref_slice = slice(npts - nbndry, npts, 1)
        off_slices = [self._shift_slice(ref_slice, offsets[k], npts) for k in range(len(offsets))]

        self._apply_to_array(yd, y, weights, off_slices, ref_slice, dim)

        h_inv = 1. / h ** deriv
        return yd * h_inv

    def diff_non_uni(self, y, coords, dim, coefs):
        """The core function to take a partial derivative on a non-uniform grid"""

        yd = np.zeros_like(y)

        ndims = len(y.shape)
        multi_slice = [slice(None, None)] * ndims
        ref_multi_slice = [slice(None, None)] * ndims

        for i, x in enumerate(coords):
            weights = coefs[i]["coefficients"]
            offsets = coefs[i]["offsets"]
            ref_multi_slice[dim] = i

            for off, w in zip(offsets, weights):
                multi_slice[dim] = i + off
                yd[ref_multi_slice] += w * y[multi_slice]

        return yd

    def _apply_to_array(self, yd, y, weights, off_slices, ref_slice, dim):
        """Applies the finite differences only to slices along a given axis"""

        ndims = len(y.shape)

        all = slice(None, None, 1)

        ref_multi_slice = [all] * ndims
        ref_multi_slice[dim] = ref_slice

        for w, s in zip(weights, off_slices):
            off_multi_slice = [all] * ndims
            off_multi_slice[dim] = s
            if abs(1 - w) < 1.E-14:
                yd[ref_multi_slice] += y[off_multi_slice]
            else:
                yd[ref_multi_slice] += w * y[off_multi_slice]

    def _shift_slice(self, sl, off, max_index):

        if sl.start + off < 0 or sl.stop + off > max_index:
            raise IndexError("Shift slice out of bounds")

        return slice(sl.start + off, sl.stop + off, sl.step)


class Coef(object):
    """
        Encapsulates a constant (number) or variable (N-dimensional coordinate array) value to multiply with a linear operator

        :param value: a number or an numpy.ndarray with meshed coordinates

        ============
        **Example**:

           The following example defines the differential operator

           .. math::

              2x \frac{\partial^3}{\partial x^2 \partial z}

           >>> X, Y, Z, U = numpy.meshgrid(x, y, z, u, indexing="ij")
           >>> diff_op = Coef(2*X) * FinDiff((0, dx, 2), (2, dz, 1))


        """
    def __init__(self, value):
        self.value = value

# Alias for backward compatibility
Coefficient = Coef

class Identity(FinDiff):

    def __init__(self):
        super().__init__()
        self.spac = 0


class PartialDerivative(UnaryOperator):

    def __init__(self, *args, **kwargs):
        """ Representation of a general partial derivative 

                \frac{\partial^(n_i + n_j + ... + n_k) / \partial}
                     {\partial x_i^n_i \partial x_j^n_j ... \partial x_k^_k}

            args:
            -----
                   A list of tuples of the form
                         (axis, M, derivative order)

                      where M is the grid spacing for uniform grids or the 1D-array of coordinates for non-uniform grids.

                   If the list contained only one tuple, you can skip the tuple parentheses.

                   An empty argument list is equivalent to the identity operator.

         """

        tuples = self._convert_to_valid_tuple_list(args)
        self.derivs = {}
        self.spac = {}
        self.coords = {}
        for t in tuples:
            axis, spac_or_coords, order = t
            if axis in self.derivs:
                raise ValueError("Derivative along axis %d specified more than once." % axis)
            self.derivs[axis] = order

            if hasattr(spac_or_coords, "__len__"):
                self.coords[axis] = spac_or_coords
            else:
                self.spac[axis] = spac_or_coords

    def axes(self):
        return sorted(list(self.derivs.keys()))

    def order(self, axis):
        if axis in self.derivs:
            return self.derivs[axis]
        return 0

    def apply(self, fd, u):

        for axis, order in self.derivs.items():
            if self.spac:
                u = fd.diff(u, self.spac[axis], order, axis, coefficients(order, fd.acc))
            else:
                coefs = []
                for i in range(len(self.coords[axis])):
                    coefs.append(coefficients_non_uni(order, fd.acc, self.coords[axis], i))
                u = fd.diff_non_uni(u, self.coords[axis], axis, coefs)

        return u

    def _convert_to_valid_tuple_list(self, args):

        all_are_tuples = True
        for arg in args:
            if not isinstance(arg, tuple):
                all_are_tuples = False
                break

        if all_are_tuples:
            all_tuples = list(args)
            for i, t in enumerate(all_tuples):
                if len(t) == 2:
                    all_tuples[i] = (t[0], t[1], 1)
        else:

            if len(args) == 2:
                all_tuples = [(args[0], args[1], 1)]
            else:
                all_tuples = [tuple(args)]

        for t in all_tuples:
            self._assert_tuple_valid(t)

        return all_tuples

    def _assert_tuple_valid(self, t):

        if len(t) > 3:
            raise ValueError("Too many arguments in tuple.")
        axis, coords_or_spac, order = t
        if not isinstance(axis, int) or axis < 0:
            raise ValueError("Axis must be non-negative integer.")
        if not hasattr(coords_or_spac, "__len__"):
            h = coords_or_spac
            if h <= 0:
                raise ValueError("Spacing must be greater than zero.")
        if not isinstance(order, int) or order <= 0:
            raise ValueError("Derivative order must be positive integer.")


