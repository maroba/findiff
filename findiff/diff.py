import numpy as np
from findiff.coefs import coefficients, coefficients_non_uni


class FinDiff(object):
    """Wrapper class for finite difference linear differential operators in any number of dimensions
       on uniform and non-uniform grids. 
       
       FinDiff objects can be added with other FinDiff objects. They can be multiplied by
       objects of type Coefficient."""

    def __init__(self, *args, **kwargs):
        """Constructor for FinDiff class
        
            Parameters:
            -----------
            
            args        list of tuples of the form 
            
                               (axis, spacing, count)     for uniform grids
                               (axis, count)              for non-uniform grids.
                               
                        "axis" is the dimension along which to take derivative. 
                        "spacing" is the grid spacing of the uniform grid along that axis.                        
                        "count" is the order of the derivative, which is optional an defaults to 1. 
                        
                        If only one tuple is given, you can leave away the tuple parentheses.
            
            **kwargs:
                            
                coords  A list of 1D-arrays of real numbers with the coordinate values along each axis.
                        This must be given to use a non-uniform grid.
                        
                acc     even integer
                        The desired accuracy order. Default is acc=2."""

        self.uniform = True
        if "coords" in kwargs:
            coords = kwargs.pop("coords")
            self.uniform = False

        if "empty" in kwargs and kwargs["empty"]:
            self._basic_ops = []
            self._coefs = []
        else:
            if self.uniform:
                self._basic_ops = [BasicFinDiff(*args, **kwargs)]
            else:
                self._basic_ops = [BasicFinDiffNonUniform(coords, *args, **kwargs)]

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
        """Add FinDiff object with other FinDiff object to linear combination.
        
           Both FinDiff objects must use the same grid.
        """
        if self._grids_are_incompatible(other):
            raise ValueError("Operators on incompatible grids cannot be added.")

        new_op = FinDiff(empty=True)
        new_op.uniform = self.uniform
        new_op._basic_ops.extend(self._basic_ops)
        new_op._coefs.extend(self._coefs)
        new_op._basic_ops.extend(other._basic_ops)
        new_op._coefs.extend(other._coefs)
        return new_op

    def __mul__(self, other):
        """Multiply FinDiff object with object of type Coefficient."""

        if not isinstance(other, Coefficient):
            other = Coefficient(other)

        return other * self

    def __rmul__(self, other):
        """Multiply FinDiff object with object of type Coefficient."""

        if not isinstance(other, Coefficient):
            other = Coefficient(other)

        new_op = FinDiff(empty=True)
        new_op.uniform = self.uniform
        new_op._basic_ops.extend(self._basic_ops)
        new_op._coefs.extend(self._coefs)

        for i in range(len(new_op._coefs)):
            new_op._coefs[i].value *= other.value

        return new_op

    def _grids_are_incompatible(self, other):
        if self.uniform != other.uniform:
            return True
        if not self.uniform and not other.uniform:
            coords1 = self._basic_ops[0]._coords
            coords2 = other._basic_ops[0]._coords
            return (coords1 != coords2).any()
        return False


class FinDiffMixIn(object):

    def _apply_to_array(self, yd, y, weights, off_slices, ref_slice, dim):
        """Applies the finite differences only to slices along a given axis"""

        ndims = len(y.shape)

        all = slice(None, None, 1)

        ref_multi_slice = [all] * ndims
        ref_multi_slice[dim] = ref_slice

        for w, s in zip(weights, off_slices):
            off_multi_slice = [all] * ndims
            off_multi_slice[dim] = s
            yd[ref_multi_slice] += w * y[off_multi_slice]

    def _shift_slice(self, sl, off, max_index):

        if sl.start + off < 0 or sl.stop + off > max_index:
            raise IndexError("Shift slice out of bounds")

        return slice(sl.start + off, sl.stop + off, sl.step)

    def _wrap_in_array(self, val):

        if hasattr(val, "__len__"):
            return np.array(val)

        return np.array([val])


class BasicFinDiff(FinDiffMixIn):
    """A basic partial derivative of any order and accuracy on a uniform grid. 
       Should not be instantiated directly, use the FinDiff class instead.
    """

    def __init__(self, *args, **kwargs):
        """Constructor for BasicFinDiff class

                    Parameters:
                    -----------

                    args        list of tuples of the form 

                                       (axis, spacing, count)     for uniform grids

                                "axis" is the dimension along which to take derivative. 
                                "spacing" is the grid spacing of the uniform grid along that axis.                        
                                "count" is the order of the derivative, which is optional an defaults to 1. 

                                If only one tuple is given, you can leave away the tuple parentheses.

                    **kwargs:

                        acc     even integer
                                The desired accuracy order. Default is acc=2."""

        self.derivs = self._parse_args(args)

        kws = set(kwargs.keys())

        self.acc = 2

        for kw in kws:
            arg = kwargs.pop(kw)
            if kw == "acc":
                if not isinstance(arg, int) or arg < 2:
                    raise ValueError("acc must be integer >= 2")
                self.acc = arg

        if kwargs:
            raise ValueError("Invalid kwargs: {}".format(kwargs))

        self._determine_coefs()

    def _parse_args(self, args):

        result = {}

        no_args_are_tuple = True
        for arg in args:
            if hasattr(arg, "__len__"):
                no_args_are_tuple = False
                break

        if no_args_are_tuple:
            if len(args) == 2:
                axis, h = args
                order = 1
            elif len(args) == 3:
                axis, h, order = args
            else:
                raise TypeError("Arguments must be tuple-like: (axis, spacing, [deriv_order])")

            result[axis] = {"h": h, "order": order}

        else:

            for arg in args:

                if not hasattr(arg, "__len__"): # we expect a non-mixed derivative
                    raise TypeError("Arguments must be tuple-like: (axis, spacing, [deriv_order])")
                if not 1 < len(arg) < 4:
                    raise ValueError("Arguments must be tuple-like: (axis, spacing, [deriv_order])")
                if len(arg) == 3:
                    axis, h, order = arg
                else:
                    axis, h = arg
                    order = 1
                if not isinstance(axis, int):
                    raise TypeError("axis arg must be integer")
                if axis < 0:
                    raise ValueError("axis arg must be >= 0")
                if hasattr(h, "__len__"):
                    raise TypeError("spacing must be scalar")
                if h <= 0:
                    raise ValueError("spacing arg must be > 0")
                if not isinstance(order, int):
                    raise TypeError("deriv order arg must be integer")
                if order <= 0:
                    raise ValueError("deriv order arg must be > 0")
                if axis in result:
                    raise ValueError("axis can only be specified once")

                result[axis] = {"h": h, "order": order}

        return result

    def _determine_coefs(self):
        """Calculates the finite difference coefficients for the requested partial derivatives"""

        for axis, partial in self.derivs.items():
            coefs = coefficients(partial["order"], self.acc)
            self.derivs[axis]["coefs"] = coefs

    def __call__(self, y):
        """Applies the finite difference operator to a function y"""

        ndims = len(y.shape)

        max_axis = max(self.derivs.keys())
        if max_axis >= ndims:
            raise IndexError(\
                "Requested derivative along axis {}, but max. axis of array is {}".format(max_axis, ndims-1))

        yd = np.array(y)

        for axis, partial in self.derivs.items():
            yd = self._diff(yd, partial["h"], partial["order"], axis, self.acc, partial["coefs"])

        return yd

    def _diff(self, y, h, deriv, dim, acc, coefs=None):
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

        h_inv = 1./h**deriv
        return yd * h_inv


class BasicFinDiffNonUniform(FinDiffMixIn):
    """A basic partial derivative of any order and accuracy on a non-uniform grid. 
       Should not be instantiated directly, use the FinDiff class instead.
    """

    def __init__(self, coords, *args, **kwargs):
        """Constructor for BasicFinDiffNonUniform class

            Parameters:
            -----------

            args        list of tuples of the form 

                               (axis, count)              for non-uniform grids.

                        "axis" is the dimension along which to take derivative. 
                        "count" is the order of the derivative, which is optional an defaults to 1. 

                        If only one tuple is given, you can leave away the tuple parentheses.

            **kwargs:

                coords  A list of 1D-arrays of real numbers with the coordinate values along each axis.
                        This must be given to use a non-uniform grid. NOT OPTIONAL!!

                acc     even integer
                        The desired accuracy order. Default is acc=2."""

        self.coords = np.array(coords)
        self.shape = [len(self.coords[k]) for k in range(len(self.coords))]
        self.derivs = self._parse_args(args)

        kws = set(kwargs.keys())

        self.acc = 2

        for kw in kws:
            arg = kwargs.pop(kw)
            if kw == "acc":
                if not isinstance(arg, int) or arg < 2:
                    raise ValueError("acc must be integer >= 2")
                self.acc = arg

        if kwargs:
            raise ValueError("Invalid kwargs: {}".format(kwargs))

        self._determine_coefs()

    def _determine_coefs(self):
        """Calculates the finite difference coefficients for the requested partial derivatives"""

        for axis, partial in self.derivs.items():
            coefs = []
            for i in range(len(self.coords[axis])):
                coefs.append(coefficients_non_uni(partial["order"], self.acc, self.coords[axis], i))

            self.derivs[axis]["coefs"] = coefs

    def _parse_args(self, args):

        result = {}

        no_args_are_tuple = True
        for arg in args:
            if hasattr(arg, "__len__"):
                no_args_are_tuple = False
                break

        if no_args_are_tuple:
            if len(args) == 1:
                axis, = args
                order = 1
            elif len(args) == 2:
                axis, order = args
            else:
                raise TypeError("Arguments must be tuple-like: (axis, spacing, [deriv_order])")

            result[axis] = {"order": order}

        else:

            for arg in args:

                if not hasattr(arg, "__len__"):
                    raise TypeError("Arguments must be tuple-like: (axis, [deriv_order])")
                if not 1 <= len(arg) <= 2:
                    raise ValueError("Arguments must be tuple-like: (axis, [deriv_order])")
                if len(arg) == 2:
                    axis, order = arg
                else:
                    axis, = arg
                    order = 1
                if not isinstance(axis, int):
                    raise TypeError("axis arg must be integer")
                if axis < 0:
                    raise ValueError("axis arg must be >= 0")
                if not isinstance(order, int):
                    raise TypeError("deriv order arg must be integer")
                if order <= 0:
                    raise ValueError("deriv order arg must be > 0")
                if axis in result:
                    raise ValueError("axis can only be specified once")

                result[axis] = {"order": order}

        return result

    def __call__(self, y):

        ndims = len(y.shape)

        max_axis = max(self.derivs.keys())
        if max_axis >= ndims:
            raise IndexError(\
                "Requested derivative along axis {}, but max. axis of array is {}".format(max_axis, ndims-1))

        if (y.shape != tuple(self.shape)):
            raise IndexError("Grid is incompatible with array to differentiate.")

        yd = np.array(y)

        for axis, partial in self.derivs.items():
            yd = self._diff(yd, self.coords, axis, partial["coefs"])

        return yd

    def _diff(self, y, coords, dim, coefs):
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


class Coefficient(object):
    """Encapsulates a constant (number) or variable (coordinate array) value to multiply with a linear operator
    """

    def __init__(self, value):
        self.value = value


