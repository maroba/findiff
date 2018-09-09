from copy import deepcopy
import numpy as np
from findiff.operators import Plus, Operator, UnaryOperator, Multiply
from findiff.coefs import coefficients, coefficients_non_uni


class FinDiff(UnaryOperator):

    def __init__(self, *args, **kwargs):
        """
            Same args as PartialDerivative class. Describes a general linear differential operator.
        
        """

        self.acc = None
        if "acc" in kwargs:
            self.acc = kwargs["acc"]
            if self.acc % 2 == 1:
                self.acc += 1

        self.root = PartialDerivative(*args)

        self.coords = None
        if "coords" in kwargs:
            self.coords = kwargs["coords"]

        self.child = None

    def __call__(self, u, **kwargs):

        for kwarg in kwargs:
            if kwarg == "spac":
                spac = kwargs[kwarg]
                if not hasattr(spac, "__getitem__"):
                    raise Exception("spac must be list or dict.")
            elif kwarg == "acc":
                self.set_accuracy(kwargs[kwarg])
            elif kwarg == "coords":
                self.coords = kwargs[kwarg]
            else:
                raise Exception("Unknown kwarg.")

        if self.acc is None:
            self.acc = 2

        if self.child is not None:
            u = self.child.apply(self, u)

        return self.root.apply(self, u)

    def set_accuracy(self, acc):
        self.acc = acc
        if self.child:
            self.child.set_accuracy(acc)

    def is_uniform(self):
        if self.coords:
            return False
        return True

    def __add__(self, other):
        fd = deepcopy(self)
        fd.root = Plus(fd.root, deepcopy(other))
        return fd

    def __rmul__(self, other):
        """
            'other' is the thing on the left side of '*'.        
        """

        if isinstance(other, Coef):
            mult = Multiply(other.value, deepcopy(self))
        else:
            mult = Multiply(other, deepcopy(self))
        fd = deepcopy(self)
        fd.root = mult

        return fd

    def __mul__(self, other):
        """Entered if self is FinDiff object in expression is self * other """

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

    def __init__(self, value):
        self.value = value

# Alias for backward compatibility
Coefficient = Coef

class Identity(FinDiff):

    def __init__(self):
        super().__init__()
        self.spac = 0


class PartialDerivative(UnaryOperator):

    def __init__(self, *args):
        """ Representation of a general partial derivative 

                \frac{\partial^(n_i + n_j + ... + n_k) / \partial}
                     {\partial x_i^n_i \partial x_j^n_j ... \partial x_k^_k}

            args:
            -----
                   A list of tuples of the form
                         (axis, derivative order)

                   If the list contained only one tuple, you can skip the tuple parentheses.

                   An empty argument list is equivalent to the identity operator.

         """

        tuples = self._convert_to_valid_tuple_list(args)
        self.derivs = {}
        self.spac = {}
        for t in tuples:
            axis, spac, order = t
            if axis in self.derivs:
                raise ValueError("Derivative along axis %d specified more than once." % axis)
            self.derivs[axis] = order
            self.spac[axis] = spac

    def axes(self):
        return sorted(list(self.derivs.keys()))

    def order(self, axis):
        if axis in self.derivs:
            return self.derivs[axis]
        return 0

    def apply(self, fd, u):

        for axis, order in self.derivs.items():
            if fd.is_uniform():
                u = fd.diff(u, self.spac[axis], order, axis, coefficients(order, fd.acc))
            else:
                coefs = []
                for i in range(len(fd.coords[axis])):
                    coefs.append(coefficients_non_uni(order, fd.acc, fd.coords[axis], i))
                u = fd.diff_non_uni(u, fd.coords[axis], axis, coefs)

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
        axis, h, order = t
        if not isinstance(axis, int) or axis < 0:
            raise ValueError("Axis must be non-negative integer.")
        if h <= 0:
            raise ValueError("Spacing must be greater than zero.")
        if not isinstance(order, int) or order <= 0:
            raise ValueError("Derivative order must be positive integer.")