import operator
import numpy as np
from findiff.coefs import coefficients


class BinaryOperator(object):

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def apply(self, rhs, *args, **kwargs):

        if isinstance(self.right, LinearMap):
            right = self.right.apply(rhs, *args, **kwargs)
        else:
            right = self.right

        if isinstance(self.left, LinearMap):
            left = self.left.apply(right, *args, **kwargs)
        else:
            left = self.left

        return self.oper(left, right)

    def __call__(self, rhs, *args, **kwargs):
        return self.apply(rhs, *args, **kwargs)


class Plus(BinaryOperator):

    def __init__(self, left, right):
        super().__init__(left, right)
        self.oper = operator.add

    def __add__(self, other):
        return Plus(self, other)

    def __radd__(self, other):
        return Plus(self, other)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(self, other)

    def __pow__(self, other):
        return Pow(self, other)

    def __rpow__(self, other):
        return Pow(self, other)

    def apply(self, rhs, *args, **kwargs):

        if isinstance(self.right, LinearMap):
            right = self.right.apply(rhs, *args, **kwargs)
        else:
            right = self.right

        if isinstance(self.left, LinearMap):
            left = self.left.apply(rhs, *args, **kwargs)
        else:
            left = self.left * rhs

        return left + right


class Mul(BinaryOperator):

    def __init__(self, left, right):
        super().__init__(left, right)
        self.oper = operator.mul

    def __add__(self, other):
        return Plus(self, other)

    def __radd__(self, other):
        return Plus(self, other)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(self, other)

    def __pow__(self, other):
        return Pow(self, other)

    def __rpow__(self, other):
        return Pow(self, other)

    def apply(self, rhs, *args, **kwargs):

        if isinstance(self.right, LinearMap):
            result = self.right.apply(rhs, *args, **kwargs)
        else:
            result = self.right * rhs

        if isinstance(self.left, LinearMap):
            result = self.left.apply(result, *args, **kwargs)
        else:
            result = self.left * result

        return result

class Pow(BinaryOperator):

    def __init__(self, left, right):
        super().__init__(left, right)
        if not isinstance(right, int) or right < 0:
            raise ValueError('Differential operators can only be raised to positive integer powers')

    def __pow__(self, power):
        return Pow(self, power)

    def __rpow__(self, power):
        return Pow(self, power)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(self, other)

    def apply(self, rhs, *args, **kwargs):

        result = rhs

        for _ in range(self.right):
            result = self.right.apply(result, *args, **kwargs)
        return result


class LinearMap(object):

    def __init__(self, value):
        self.value = value

    def __add__(self, other):
        return Plus(self, other)

    def __radd__(self, other):
        return Plus(self, other)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(self, other)

    def __pow__(self, other):
        return Pow(self, other)

    def __rpow__(self, other):
        return Pow(self, other)

    def apply(self, rhs, *args, **kwargs):
        raise NotImplementedError('Base class LinearMap is not to be used directly!')


class Diff(LinearMap):

    def __init__(self, axis, order=1):
        self.axis = axis
        self.order = order

    def __call__(self, rhs, *args, **kwargs):
        return self.apply(rhs, *args, **kwargs)

    def apply(self, u, h, **kwargs):

        if isinstance(h, dict):
            h = h[self.axis]

        return self.diff(u, h, **kwargs)

    def diff(self, y, h, **kwargs):
        """The core function to take a partial derivative on a uniform grid.

            Central coefficients will be used whenever possible. Backward or forward
            coefficients will be used if not enough points are available on either side,
            i.e. forward coefficients for the low index boundary and backward coefficients
            for the high index boundary.
        """

        acc = 2

        if "acc" in kwargs:
            acc = kwargs["acc"]

        dim = self.axis
        coefs = coefficients(self.order, acc)
        deriv = self.order

        try:
            npts = y.shape[dim]
        except AttributeError as err:
            raise ValueError(
                "FinDiff objects can only be applied to arrays or evaluated(!) functions returning arrays") from err

        scheme = "center"
        weights = coefs[scheme]["coefficients"]
        offsets = coefs[scheme]["offsets"]

        num_bndry_points = len(weights) // 2
        ref_slice = slice(num_bndry_points, npts - num_bndry_points, 1)
        off_slices = [self._shift_slice(ref_slice, offsets[k], npts) for k in range(len(offsets))]

        yd = np.zeros_like(y)

        self._apply_to_array(yd, y, weights, off_slices, ref_slice, dim)

        scheme = "forward"
        weights = coefs[scheme]["coefficients"]
        offsets = coefs[scheme]["offsets"]

        ref_slice = slice(0, num_bndry_points, 1)
        off_slices = [self._shift_slice(ref_slice, offsets[k], npts) for k in range(len(offsets))]

        self._apply_to_array(yd, y, weights, off_slices, ref_slice, dim)

        scheme = "backward"
        weights = coefs[scheme]["coefficients"]
        offsets = coefs[scheme]["offsets"]

        ref_slice = slice(npts - num_bndry_points, npts, 1)
        off_slices = [self._shift_slice(ref_slice, offsets[k], npts) for k in range(len(offsets))]

        self._apply_to_array(yd, y, weights, off_slices, ref_slice, dim)

        h_inv = 1. / h ** deriv
        return yd * h_inv

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
                yd[tuple(ref_multi_slice)] += y[tuple(off_multi_slice)]
            else:
                yd[tuple(ref_multi_slice)] += w * y[tuple(off_multi_slice)]

    def _shift_slice(self, sl, off, max_index):

        if sl.start + off < 0 or sl.stop + off > max_index:
            raise IndexError("Shift slice out of bounds")

        return slice(sl.start + off, sl.stop + off, sl.step)


class Id(LinearMap):

    def __init__(self):
        self.value = 1

    def apply(self, rhs, *args, **kwargs):
        return rhs


class Scalar(object):

    def __init__(self, value):
        self.value = value

    def __mul__(self, other):
        return Mul(self.value , other)

#    def __rmul__(self, other):
#        print("rmul of %s called" % self.name)

"""
L = Diff(0, 2) + X * Diff(0) * Diff(1) + Diff(1, 2) + X + 1
L = D(0, 2) + X * D(0) * D(1) + D(1, 2) + X + 1

L.set_properties(acc=2, h={0: dx, 1: dy})
L(u)

or 

L(u, acc=2, h={0: dx, 1: dy})

"""

#L = LinearMap(1) + 2*(LinearMap(3) + LinearMap(4) + LinearMap(5)) ** LinearMap(2) * LinearMap(6)
#L = LinearMap(3) * 2
#print(L.apply(1))



