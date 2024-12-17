import pytest

from findiff.grids import EquidistantAxis


def test_create_equidistantaxis():
    axis = EquidistantAxis(1, 0.1)
    assert axis.spacing == 0.1
    assert axis.dim == 1

    assert axis.periodic is False

    axis = EquidistantAxis(1, 0.1, periodic=True)
    assert axis.periodic is True


def test_create_equidistantaxis_fails_invalid_args():
    with pytest.raises(ValueError):
        EquidistantAxis(1, 0.0)

    with pytest.raises(ValueError):
        EquidistantAxis(-1, 0.1)

    with pytest.raises(ValueError, match="Dimension must be an integer"):
        EquidistantAxis(0.1, 1)
