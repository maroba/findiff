from itertools import product
import numpy as np


def interior_mask_as_ndarray(shape):
    ndims = len(shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[tuple([slice(1, -1)] * ndims)] = True
    return mask


def all_index_tuples_as_list(shape):
    ndims = len(shape)
    return list(product(*tuple([list(range(shape[k])) for k in range(ndims)])))


def long_indices_as_ndarray(shape):
    return np.array(list(range(np.prod(shape)))).reshape(shape)


def to_long_index(idx, shape):
    ndims = len(shape)
    long_idx = 0
    siz = 1
    for axis in range(ndims):
        long_idx += idx[ndims-1-axis] * siz
        siz *= shape[ndims-1-axis]
    return long_idx


def to_index_tuple(long_idx, shape):
    ndims = len(shape)
    idx = np.zeros(ndims)
    for k in range(ndims):
        s = np.prod(shape[k+1:])
        idx[k] = long_idx // s
        long_idx = long_idx - s * idx[k]

    return tuple(idx)




