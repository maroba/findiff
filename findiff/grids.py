from abc import ABC


class Grid(ABC):
    pass


class EquidistantGrid(Grid):

    def __init__(self, config: dict):

        self.spacing = {}
        self.periodic = {}
        for axis, spacing_or_dict in config.items():
            if isinstance(spacing_or_dict, dict):
                self.spacing[axis] = spacing_or_dict["h"]
                self.periodic[axis] = spacing_or_dict.get("periodic", False)
            else:
                self.spacing[axis] = spacing_or_dict
                self.periodic[axis] = False
            if self.spacing[axis] <= 0:
                raise ValueError("spacing must be greater than zero")


class TensorProductGrid(Grid):

    def __init__(self, axes_coords: dict):
        if len(axes_coords) == 0:
            raise ValueError("axes_coords cannot be empty")
        for axis, c in axes_coords.items():
            if len(c.shape) != 1:
                raise ValueError(
                    "each item in axes_coords must have only one dimension"
                )
        self.coords = axes_coords
