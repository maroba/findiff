import numbers

import numpy as np


class GridAxis:

    def __init__(self, dim: int, periodic=False):
        if not isinstance(dim, numbers.Number) or int(dim) != dim:
            raise ValueError("Dimension must be an integer")
        if dim < 0:
            raise ValueError("Dimension must be >= 0.")
        self.dim = dim
        self.periodic = periodic


class EquidistantAxis(GridAxis):

    def __init__(self, dim: int, spacing: float, periodic=False):
        super().__init__(dim, periodic)
        if spacing <= 0:
            raise ValueError("Spacing must be > 0.")
        self.spacing = spacing


class NonEquidistantAxis(GridAxis):
    def __init__(self, dim: int, coords: np.ndarray, periodic=False):
        super().__init__(dim, periodic)
        self.coords = coords


class Grid:
    def __init__(self, *axes: GridAxis):
        self.axes = {ax.dim: ax for ax in axes}

    def get_axis(self, dim: int) -> GridAxis:
        return self.axes.get(dim)


def make_grid(config_or_grid):
    """Makes or returns a grid based on configuration or an actual Grid instance.

    Historically, the API allowed to specify grid using a variety of
    shortcuts using a single number, dicts or full Grid instances.

    The purpose of this function is to keep other modules closed with
    respect of addition an modification of GridAxis and Grid types.
    """
    if isinstance(config_or_grid, Grid):
        return config_or_grid
    elif isinstance(config_or_grid, dict):
        config = config_or_grid
        axes = []
        for dim, ax_config in config.items():
            if isinstance(ax_config, dict):
                ax = EquidistantAxis(
                    dim, ax_config["h"], periodic=ax_config.get("periodic", False)
                )
            else:
                ax = EquidistantAxis(dim, ax_config)
            axes.append(ax)
        return Grid(*axes)
    elif config_or_grid is None:
        return None
    else:
        raise TypeError(f"Unsupported grid type: {type(config_or_grid)}")


def make_axis(dim, config_or_axis, periodic=False):
    """Makes or returns a grid axis based on configuration or an actual GridAxis instance.

    Historically, the API allowed to specify axes using a variety of
    shortcuts using a single number, dicts or full GridAxis instances.

    The purpose of this function is to keep other modules closed with
    respect of addition an modification of GridAxis and Grid types.
    """

    if isinstance(config_or_axis, GridAxis):
        return config_or_axis
    if isinstance(config_or_axis, numbers.Number):
        return EquidistantAxis(dim, spacing=config_or_axis, periodic=periodic)
    elif isinstance(config_or_axis, np.ndarray):
        return NonEquidistantAxis(dim, coords=config_or_axis, periodic=periodic)
