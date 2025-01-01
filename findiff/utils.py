import itertools
from itertools import product

import numpy as np
from scipy.sparse import diags


def interior_mask_as_ndarray(shape):
    ndims = len(shape)
    mask = np.zeros(shape, dtype=bool)
    mask[tuple([slice(1, -1)] * ndims)] = True
    return mask


def all_index_tuples_as_list(shape):
    ndims = len(shape)
    return list(product(*tuple([list(range(shape[k])) for k in range(ndims)])))


def get_long_indices_for_all_grid_points_as_ndarray(shape):
    return get_long_indices_for_all_grid_points_as_1d_array(shape).reshape(shape)


def get_long_indices_for_all_grid_points_as_1d_array(shape):
    return np.arange(np.prod(shape), dtype=np.int64)


def to_long_index(idx, shape):

    ndims = len(shape)
    long_idx = 0
    siz = 1
    for axis in range(ndims):
        idx_ = idx[ndims - 1 - axis]
        long_idx += idx_ * siz
        siz *= shape[ndims - 1 - axis]

    return long_idx


def to_index_tuple(long_idx, shape):
    ndims = len(shape)
    idx = np.zeros(ndims)
    for k in range(ndims):
        s = np.prod(shape[k + 1 :])
        idx[k] = long_idx // s
        long_idx = long_idx - s * idx[k]

    return tuple(idx)


def get_list_of_multiindex_tuples(shape):
    short_inds = [np.arange(shape[k]) for k in range(len(shape))]
    short_inds = list(itertools.product(*short_inds))
    return short_inds


def create_cyclic_band_diagonal(n, offsets, band_values):
    """
    Create a cyclic band-diagonal matrix using scipy.sparse.

    Parameters:
        n (int): The size of the matrix (n x n).
        offsets (list of int): Offsets for the bands (negative, 0, or positive).
        band_values (list of float): Values to fill in the bands (length must match num_bands).

    Returns:
        scipy.sparse.csr_matrix: Cyclic band-diagonal matrix.
    """
    num_bands = len(offsets)
    if len(offsets) != num_bands or len(band_values) != num_bands:
        raise ValueError(
            "Offsets and band_values must match the number of bands (num_bands)."
        )

    # Create the diagonal values for each band
    diagonals = []
    for offset, value in zip(offsets, band_values):
        diag = np.full(n, value)
        diagonals.append(diag)

    # Build the band-diagonal matrix
    band_matrix = diags(diagonals, offsets, shape=(n, n), format="csr")

    # Add cyclic wrap-around connections
    for offset, value in zip(offsets, band_values):
        if offset > 0:  # Wrap from top rows to bottom
            band_matrix += diags(
                [np.full(offset, value)], [offset - n], shape=(n, n), format="csr"
            )
        elif offset < 0:  # Wrap from bottom rows to top
            band_matrix += diags(
                [np.full(-offset, value)], [offset + n], shape=(n, n), format="csr"
            )

    return band_matrix
