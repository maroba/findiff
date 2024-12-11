from abc import ABC


class Grid(ABC):
    pass


class EquidistantGrid(Grid):

    def __init__(self, spacing: dict):
        for axis, h in spacing.items():
            if h <= 0:
                raise ValueError("spacing must be greater than zero")

        # key -> value == axis -> spacing for axis
        self.spacing = spacing


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
