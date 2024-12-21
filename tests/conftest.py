import numpy as np
import pytest


@pytest.fixture()
def grid_data(request):
    m = request.node.get_closest_marker("grid_spec")
    if m is None:
        raise ValueError("Grid data marker not defined")
    shape = m.kwargs["shape"]
    edges = m.kwargs["edges"]
    endpoints = m.kwargs.get("endpoints", [True] * len(shape))

    if len(shape) == 1:
        x = np.linspace(*edges, shape[0], endpoint=endpoints[0])
        dx = x[1] - x[0]
        return x, dx

    axes = tuple(
        [
            np.linspace(edges[k][0], edges[k][1], shape[k], endpoint=endpoints[k])
            for k in range(len(shape))
        ]
    )
    coords = np.meshgrid(*axes, indexing="ij")
    spacings = [axes[k][1] - axes[k][0] for k in range(len(shape))]
    return axes, spacings, coords
