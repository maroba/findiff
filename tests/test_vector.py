import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from findiff import Gradient, Divergence, Curl, Laplacian


class TestGradient:

    def test_3d_gradient_on_scalar_func(self):
        axes, h, [X, Y, Z] = init_mesh(3, (50, 50, 50))
        f = np.sin(X) * np.sin(Y) * np.sin(Z)
        grad_f_ex = np.array(
            [
                np.cos(X) * np.sin(Y) * np.sin(Z),
                np.sin(X) * np.cos(Y) * np.sin(Z),
                np.sin(X) * np.sin(Y) * np.cos(Z),
            ]
        )
        grad = Gradient(spac=h, acc=4)
        grad_f = grad(f)
        assert_array_almost_equal(grad_f, grad_f_ex)

    def test_spacing_with_h(self):
        axes, h, [X, Y, Z] = init_mesh(3, (50, 50, 50))
        f = np.sin(X) * np.sin(Y) * np.sin(Z)
        grad_f_ex = np.array(
            [
                np.cos(X) * np.sin(Y) * np.sin(Z),
                np.sin(X) * np.cos(Y) * np.sin(Z),
                np.sin(X) * np.sin(Y) * np.cos(Z),
            ]
        )
        grad = Gradient(h=h, acc=4)
        grad_f = grad(f)
        assert_array_almost_equal(grad_f, grad_f_ex)

    def test_3d_gradient_on_scalar_func_non_uni(self):
        axes, h, [X, Y, Z] = init_mesh(3, (50, 50, 50))
        f = np.sin(X) * np.sin(Y) * np.sin(Z)
        grad_f_ex = np.array(
            [
                np.cos(X) * np.sin(Y) * np.sin(Z),
                np.sin(X) * np.cos(Y) * np.sin(Z),
                np.sin(X) * np.sin(Y) * np.cos(Z),
            ]
        )
        grad = Gradient(coords=axes, acc=4)
        grad_f = grad(f)
        assert_array_almost_equal(grad_f, grad_f_ex)

    def test_3d_gradient_on_vector_func_should_fail(self):
        axes, h, [X, Y, Z] = init_mesh(3, (50, 50, 50))
        f = np.array(
            [np.sin(X) * np.sin(Y) * np.sin(Z), np.sin(X) * np.sin(Y) * np.sin(Z)]
        )
        grad = Gradient(spac=h, acc=4)
        with pytest.raises(ValueError):
            grad(f)


class TestDivergence:

    def test_3d_divergence_on_vector_func(self):
        axes, h, [X, Y, Z] = init_mesh(3, (50, 50, 50))
        f = np.array([np.sin(X) * np.sin(Y) * np.sin(Z)] * 3)
        assert f.shape == (3, 50, 50, 50)
        div_f_ex = (
            np.cos(X) * np.sin(Y) * np.sin(Z)
            + np.sin(X) * np.cos(Y) * np.sin(Z)
            + np.sin(X) * np.sin(Y) * np.cos(Z)
        )

        div = Divergence(spac=h, acc=4)
        div_f = div(f)
        assert_array_almost_equal(div_f, div_f_ex)

    def test_3d_divergence_on_vector_func_of_wrong_dim(self):
        axes, h, [X, Y, Z] = init_mesh(3, (50, 50, 50))
        f = np.array([np.sin(X) * np.sin(Y) * np.sin(Z)] * 3)
        assert f.shape == (3, 50, 50, 50)
        div = Divergence(spac=[1, 1], acc=4)
        with pytest.raises(ValueError):
            div(f)


class TestCurl:

    def test_curl_on_3d_vector_func(self):
        axes, h, [X, Y, Z] = init_mesh(3, (50, 50, 50))
        f = np.array([np.sin(X) * np.sin(Y) * np.sin(Z)] * 3)
        assert f.shape == (3, 50, 50, 50)
        curl_f_ex = np.array(
            [
                np.sin(X) * np.cos(Y) * np.sin(Z) - np.sin(X) * np.sin(Y) * np.cos(Z),
                np.sin(X) * np.sin(Y) * np.cos(Z) - np.cos(X) * np.sin(Y) * np.sin(Z),
                np.cos(X) * np.sin(Y) * np.sin(Z) - np.sin(X) * np.cos(Y) * np.sin(Z),
            ]
        )
        curl = Curl(spac=h, acc=4)
        curl_f = curl(f)
        assert_array_almost_equal(curl_f, curl_f_ex)

    def test_curl_for_2d(self):
        axes, h, [X, Y] = init_mesh(2, (50, 50))
        f = np.array(
            [
                -np.sin(Y) * np.cos(X),
                np.sin(X) * np.cos(Y),
            ]
        )
        # scalar curl in 2D: dF_y/dx - dF_x/dy = cos(X)*cos(Y) - (-cos(Y)*cos(X)) = 2*cos(X)*cos(Y)
        curl_f_ex = 2 * np.cos(X) * np.cos(Y)
        curl = Curl(spac=h, acc=4)
        curl_f = curl(f)
        assert_array_almost_equal(curl_f, curl_f_ex, decimal=4)

    def test_curl_for_2d_function(self):
        """2D curl of an irrotational field should be zero."""
        axes, h, [X, Y] = init_mesh(2, (50, 50))
        # gradient of a scalar is irrotational
        f = np.array([2 * X, 2 * Y])  # grad(x^2 + y^2)
        curl = Curl(spac=h, acc=4)
        curl_f = curl(f)
        assert_array_almost_equal(curl_f, np.zeros_like(X), decimal=4)

    def test_curl_for_1d(self):
        with pytest.raises(ValueError):
            Curl(spac=[1], acc=4)

    def test_curl_for_4d(self):
        with pytest.raises(ValueError):
            Curl(spac=[1, 1, 1, 1], acc=4)

    def test_curl_2d_wrong_shape(self):
        axes, h, [X, Y] = init_mesh(2, (50, 50))
        f = np.array([np.sin(X) * np.sin(Y)] * 3)  # 3 components but 2D grid
        curl = Curl(spac=h, acc=4)
        with pytest.raises(ValueError):
            curl(f)

    def test_curl_3d_wrong_shape(self):
        axes, h, [X, Y, Z] = init_mesh(3, (50, 50, 50))
        f = np.array([np.sin(X) * np.sin(Y) * np.sin(Z)] * 2)  # 2 components but 3D
        curl = Curl(spac=h, acc=4)
        with pytest.raises(ValueError):
            curl(f)

    def test_curl_2d_non_uniform(self):
        axes, h, [X, Y] = init_mesh(2, (50, 50))
        f = np.array(
            [
                -np.sin(Y) * np.cos(X),
                np.sin(X) * np.cos(Y),
            ]
        )
        curl_f_ex = 2 * np.cos(X) * np.cos(Y)
        curl = Curl(coords=axes, acc=4)
        curl_f = curl(f)
        assert_array_almost_equal(curl_f, curl_f_ex, decimal=4)

    def test_curl_2d_returns_scalar_shape(self):
        axes, h, [X, Y] = init_mesh(2, (50, 50))
        f = np.array([X, Y])
        curl = Curl(spac=h, acc=4)
        curl_f = curl(f)
        assert curl_f.shape == X.shape

    def test_laplacian(self):
        axes, h, [X, Y] = init_mesh(2, (50, 50))
        f = X**2 + Y**2
        laplace = Laplacian(h)
        np.testing.assert_array_almost_equal(laplace(f), 4 * np.ones_like(X))


def init_mesh(ndims, npoints):
    axes = [np.linspace(-1, 1, npoints[k]) for k in range(ndims)]
    h = [x[1] - x[0] for x in axes]
    mesh = np.meshgrid(*axes, indexing="ij")
    return axes, h, mesh
