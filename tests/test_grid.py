import pytest

from findiff.grids import EquidistantAxis, make_grid, make_axis


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


def test_makegrid_with_none_returns_none():
    assert make_grid(None) is None


def test_makegrid_with_invalid_arg_raises_exception():
    with pytest.raises(TypeError):
        make_grid("sndljas")


def test_makeaxis_with_gridaxis_returns_axis():
    axis = EquidistantAxis(1, 1)
    assert make_axis(1, axis) is axis
