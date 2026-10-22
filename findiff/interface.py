from .compact import CompactScheme
from .grids import make_axis
from .operators import Diff as _Diff


class Diff(_Diff):

    def __init__(
        self,
        axis=0,
        grid=None,
        periodic=False,
        acc=_Diff.DEFAULT_ACC,
        scheme=None,
        compact=None,
        spectral=False,
        boundary="one-sided",
    ):
        """Represents a partial derivative (along one axis).

        For higher derivatives, exponentiate. For mixed partial derivatives, multiply. See
        examples below.

        Parameters
        ----------
        axis : int
            The 0-based index of the axis along which to differentiate.
        grid : float, array or GridAxis, optional
            The grid along `axis`: a number is the (constant) grid spacing, a 1D
            array the coordinates of a non-uniform grid. Can also be set later
            with :meth:`set_grid`.
        periodic : bool
            Whether the grid is periodic along `axis`.
        acc : int
            The (minimum) accuracy order, a positive even number. Default 2.
        scheme : CompactScheme, optional
            Use this compact (implicit) finite difference scheme.
        compact : int, optional
            Use a compact scheme with this (odd) number of left-hand side points,
            determined from `acc` (see the guide on compact finite differences).
        spectral : bool
            With `compact`, use a scheme with spectral-like resolution instead of
            maximal accuracy order.
        boundary : str
            How points near a non-periodic boundary are treated on equidistant
            grids: ``"one-sided"`` (default) uses one-sided stencils of order
            `acc`; ``"centered"`` uses the widest centered stencil that fits.
            The latter is less accurate near the boundary but keeps
            time-dependent problems stable for high `acc`.

        Examples
        --------
        Set up grid (equidistant here):
            >>> import numpy as np
            >>> x = np.linspace(0, 10, 100)

        The array to differentiate
            >>> f = np.sin(x) # as an example

        Define the first derivative:
            >>> from findiff import Diff
            >>> d_dx = Diff(0)
            >>> d_dx.set_grid({0: x[1] - x[0]})

        Now apply it:
            >>> df_dx = d_dx(f)

        The second derivative is constructed by exponentiation:
            >>> d2_dx2 = d_dx ** 2
            >>> d2f_dx2 = d2_dx2(f)

        In multiple dimensions with meshed grids, the usage is accordingly:
            >>> x = y = z = np.linspace(0, 10, 100)
            >>> dx = dy = dz = x[1] - x[0]
            >>> X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            >>> f = np.sin(X) * np.sin(Y) * np.sin(Z)
            >>> d_dx = Diff(0)
            >>> d_dy = Diff(1)
            >>> d_dz = Diff(2)
            >>> d3_dxdydz = d_dx * d_dy * d_dz
            >>> d3_dxdydz.set_grid({0: dx, 1: dy, 2: dz})
            >>> d3f_dxdydz = d3_dxdydz(f)
        """
        if compact and scheme:
            raise ValueError("Cannot specify both `compact` and `scheme`")
        if spectral and not compact:
            raise ValueError("`spectral=True` requires `compact`")
        if compact:
            if compact % 2 == 0:
                raise ValueError("`compact` must be odd integer")
            if spectral:
                scheme = CompactScheme.spectral(1, compact, compact + 2, acc=acc)
            else:
                scheme = CompactScheme.from_accuracy(acc, 1, compact)

        grid_axis = make_axis(axis, grid, periodic)
        super().__init__(axis, grid_axis, acc, scheme, boundary=boundary)
