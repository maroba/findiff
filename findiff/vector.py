"""A module for the common differential operators of vector calculus"""

import numpy as np
from .operators import FinDiff
from.diff import Diff


class VectorOperator(object):
    """Base class for all vector differential operators.
       Shall not be instantiated directly, but through the child classes.
    """

    def __init__(self, **kwargs):
        """Constructor for the VectorOperator base class.
        
            kwargs:
            -------
            
            h       list with the grid spacings of an N-dimensional uniform grid
            
            coords  list of 1D arrays with the coordinate values along the N axes.
                    This is used for non-uniform grids. 
            
            Either specify "h" or "coords", not both.
        
        """

        if "acc" in kwargs:
            self.acc = kwargs.pop("acc")
        else:
            self.acc = 2

        if "spac" in kwargs or "h" in kwargs: # necessary for backward compatibility 0.5.2 => 0.6
            if "spac" in kwargs:
                kw = "spac"
            else:
                kw = "h"
            self.h = kwargs.pop(kw)
            self.ndims = len(self.h)
            self.components = [FinDiff(k, self.h[k], 1) for k in range(self.ndims)]

        if "coords" in kwargs:
            coords = kwargs.pop("coords")
            self.ndims = self.__get_dimension(coords)
            self.components = [FinDiff((k, coords[k], 1), **kwargs) for k in range(self.ndims)]

    def __get_dimension(self, coords):
        if isinstance(coords, np.ndarray):
            shape = coords.shape
            if len(shape) > 1:
                ndims = shape[0]
            else:
                ndims = 1
        else:
            ndims = len(coords)
        return ndims


class Gradient(VectorOperator):
    r"""
    The N-dimensional gradient.
    
    .. math::
        \nabla = \left(\frac{\partial}{\partial x_0}, \frac{\partial}{\partial x_1}, ... , \frac{\partial}{\partial x_{N-1}}\right)

    :param kwargs:  exactly one of *h* and *coords* must be specified
    
             *h* 
                     list with the grid spacings of an N-dimensional uniform grid     
             *coords*
                     list of 1D arrays with the coordinate values along the N axes.
                     This is used for non-uniform grids.
                     
             *acc*
                     accuracy order, must be positive integer, default is 2
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, f):
        """
        Applies the N-dimensional gradient to the array f.
        
        :param f:  ``numpy.ndarray``
        
                Array to apply the gradient to. It represents a scalar function,
                so it must have N axes for the N independent variables.        
           
        :returns: ``numpy.ndarray``
         
                The gradient of f, which has N+1 axes, i.e. it is 
                an array of N arrays of N axes each.
           
        """

        if not isinstance(f, np.ndarray):
            raise TypeError("Function to differentiate must be numpy.ndarray")

        if len(f.shape) != self.ndims:
            raise ValueError("Gradients can only be applied to scalar functions")

        result = []
        for k in range(self.ndims):
            d_dxk = self.components[k]
            result.append(d_dxk(f, acc=self.acc))

        return np.array(result)


class Divergence(VectorOperator):
    r"""
    The N-dimensional divergence.
    
    .. math::
    
       {\rm \bf div} = \nabla \cdot
    
    :param kwargs:  exactly one of *h* and *coords* must be specified

         *h* 
                 list with the grid spacings of an N-dimensional uniform grid     
         *coords*
                 list of 1D arrays with the coordinate values along the N axes.
                 This is used for non-uniform grids.
                 
         *acc*
                 accuracy order, must be positive integer, default is 2
    
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, f):
        """
        Applies the divergence to the array f.

        :param f: ``numpy.ndarray``
                
               a vector function of N variables, so its array has N+1 axes.
        
        :returns: ``numpy.ndarray``
            
               the divergence, which is a scalar function of N variables, so it's array dimension has N axes
                
        """
        if not isinstance(f, np.ndarray) and not isinstance(f, list):
            raise TypeError("Function to differentiate must be numpy.ndarray or list of numpy.ndarrays")

        if len(f.shape) != self.ndims + 1 and f.shape[0] != self.ndims:
            raise ValueError("Divergence can only be applied to vector functions of the same dimension")

        result = np.zeros(f.shape[1:])

        for k in range(self.ndims):
            result += self.components[k](f[k], acc=self.acc)

        return result


class Curl(VectorOperator):
    r"""
    The curl operator. 
    
    .. math::
    
        {\rm \bf rot} = \nabla \times
    
    Is only defined for 3D.
    
    :param kwargs:  exactly one of *h* and *coords* must be specified

     *h* 
             list with the grid spacings of a 3-dimensional uniform grid     
     *coords*
             list of 1D arrays with the coordinate values along the 3 axes.
             This is used for non-uniform grids.
             
     *acc*
             accuracy order, must be positive integer, default is 2

    
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.ndims != 3:
            raise ValueError("Curl operation is only defined in 3 dimensions. {} were given.".format(self.ndims))

    def __call__(self, f):
        """
        Applies the curl to the array f.

        :param f: ``numpy.ndarray``

               a vector function of N variables, so its array has N+1 axes.

        :returns: ``numpy.ndarray``

               the curl, which is a vector function of N variables, so it's array dimension has N+1 axes

        """

        if not isinstance(f, np.ndarray) and not isinstance(f, list):
            raise TypeError("Function to differentiate must be numpy.ndarray or list of numpy.ndarrays")

        if len(f.shape) != self.ndims + 1 and f.shape[0] != self.ndims:
            raise ValueError("Curl can only be applied to vector functions of the three dimensions")

        result = np.zeros(f.shape)

        result[0] += self.components[1](f[2], acc=self.acc) - self.components[2](f[1], acc=self.acc)
        result[1] += self.components[2](f[0], acc=self.acc) - self.components[0](f[2], acc=self.acc)
        result[2] += self.components[0](f[1], acc=self.acc) - self.components[1](f[0], acc=self.acc)

        return result


class Laplacian(object):
    r"""
        The N-dimensional Laplace operator.

        .. math::

           {\rm \bf \nabla^2} = \sum_{k=0}^{N-1} \frac{\partial^2}{\partial x_k^2}

        :param kwargs:  exactly one of *h* and *coords* must be specified

             *h* 
                     list with the grid spacings of an N-dimensional uniform grid     
             *coords*
                     list of 1D arrays with the coordinate values along the N axes.
                     This is used for non-uniform grids.

             *acc*
                     accuracy order, must be positive integer, default is 2

        """

    """A representation of the Laplace operator in arbitrary dimensions using finite difference schemes"""

    def __init__(self, h=[1.], acc=2):
        h = wrap_in_ndarray(h)

        self._parts = [FinDiff((k, h[k], 2), acc=acc) for k in range(len(h))]

    def __call__(self, f):
        """
        Applies the Laplacian to the array f.

        :param f: ``numpy.ndarray``

               a scalar function of N variables, so its array has N axes.

        :returns: ``numpy.ndarray``

               the Laplacian of f, which is a scalar function of N variables, so it's array dimension has N axes

        """
        laplace_f = np.zeros_like(f)

        for part in self._parts:
            laplace_f += part(f)

        return laplace_f


def wrap_in_ndarray(value):
    """Wraps the argument in a numpy.ndarray.

       If value is a scalar, it is converted in a list first.
       If value is array-like, the shape is conserved.

    """

    if hasattr(value, "__len__"):
        return np.array(value)
    else:
        return np.array([value])
