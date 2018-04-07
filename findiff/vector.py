"""A module for the common differential operators of vector calculus"""

import numpy as np
from findiff.diff import FinDiff
from findiff.util import wrap_in_ndarray


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

        if "h" in kwargs:
            self.h = wrap_in_ndarray(kwargs.pop("h"))
            self.ndims = len(self.h)
            self.components = [FinDiff((k, self.h[k]), **kwargs) for k in range(self.ndims)]

        if "coords" in kwargs:
            coords = kwargs.pop("coords")
            self.ndims = self.__get_dimension(coords)
            self.components = [FinDiff((k,), coords=coords, **kwargs) for k in range(self.ndims)]

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
    """The N-dimensional gradient
    
        (\frac{\partial}{\partial x_0}, \frac{\partial}{\partial x_1}, ... , \frac{\partial}{\partial x_{N-1}})
    
    """

    def __init__(self, **kwargs):
        """Constructor for the N-dimensional gradient

                kwargs:
                -------

                h       list with the grid spacings of an N-dimensional uniform grid

                coords  list of 1D arrays with the coordinate values along the N axes.
                        This is used for non-uniform grids. 

                      !! Either specify "h" or "coords", not both. !!

                acc     accuracy order, must be even, positive integer

        """

        super().__init__(**kwargs)

    def __call__(self, f):
        """Applies the N-dimensional gradient to the array f.
        
           Parameters:
           -----------
           
           f    array to apply the gradient to. It represents a scalar function,
                so its dimension must be N for the N independent variables.        
           
           Returns:
            
           The gradient of f, which has dimension N+1, i.e. it is 
           an array of N arrays of N dimensions each.
           
        """

        if not isinstance(f, np.ndarray):
            raise TypeError("Function to differentiate must be numpy.ndarray")

        if len(f.shape) != self.ndims:
            raise ValueError("Gradients can only be applied to scalar functions")

        result = []
        for k in range(self.ndims):
            d_dxk = self.components[k]
            result.append(d_dxk(f))

        return np.array(result)


class Divergence(VectorOperator):
    """The N-dimensional divergence
    
       \sum_{k=1}^N \frac{\partial }{\partial x_k}
    
    """

    def __init__(self, **kwargs):
        """Constructor for the N-dimensional divergence

                kwargs:
                -------

                h       list with the grid spacings of an N-dimensional uniform grid

                coords  list of 1D arrays with the coordinate values along the N axes.
                        This is used for non-uniform grids. 

                      !! Either specify "h" or "coords", not both. !!

                acc     accuracy order, must be even, positive integer

        """

        super().__init__(**kwargs)

    def __call__(self, f):
        """Applies the divergence to the array f.
        
            f is a vector function of N variables, so its array dimension is N+1.
            
            Returns the divergence, which is a scalar function of N variables, so it's array dimension is N
                
        """
        if not isinstance(f, np.ndarray) and not isinstance(f, list):
            raise TypeError("Function to differentiate must be numpy.ndarray or list of numpy.ndarrays")

        if len(f.shape) != self.ndims + 1 and f.shape[0] != self.ndims:
            raise ValueError("Divergence can only be applied to vector functions of the same dimension")

        result = np.zeros(f.shape[1:])

        for k in range(self.ndims):
            result += self.components[k](f[k])

        return result


class Curl(VectorOperator):
    """The curl operator. Is only defined for 3D."""

    def __init__(self, **kwargs):
        """Constructor for the 3-dimensional curl

                kwargs:
                -------

                h       list with the grid spacings of an 3-dimensional uniform grid

                coords  list of 1D arrays with the coordinate values along the 3 axes.
                        This is used for non-uniform grids. 

                      !! Either specify "h" or "coords", not both. !!

                acc     accuracy order, must be even, positive integer

        """

        super().__init__(**kwargs)

        if self.ndims != 3:
            raise ValueError("Curl operation is only defined in 3 dimensions. {} were given.".format(self.ndims))

    def __call__(self, f):
        """Applies the curl operator to the vector function f, represented by array of dimension 4.
        
           Returns the curl, a vector function, i.e. an array of dimension 4.
        """

        if not isinstance(f, np.ndarray) and not isinstance(f, list):
            raise TypeError("Function to differentiate must be numpy.ndarray or list of numpy.ndarrays")

        if len(f.shape) != self.ndims + 1 and f.shape[0] != self.ndims:
            raise ValueError("Curl can only be applied to vector functions of the three dimensions")

        result = np.zeros(f.shape)

        result[0] += self.components[1](f[2]) - self.components[2](f[1])
        result[1] += self.components[2](f[0]) - self.components[0](f[2])
        result[2] += self.components[0](f[1]) - self.components[1](f[0])

        return result


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

        h = wrap_in_ndarray(h)

        self._parts = [FinDiff((k, h[k], 2), acc=acc) for k in range(len(h))]

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

