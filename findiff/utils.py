import itertools
from itertools import product

import numpy as np


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


#
# The following is working, but unused yet. Commented because there are no tests yet.
#
# def deprecated(reason="This feature is deprecated."):
#     def decorator(func_or_class):
#         if isinstance(func_or_class, type):  # Handle classes
#             original_init = func_or_class.__init__
#
#             @wraps(original_init)
#             def new_init(self, *args, **kwargs):
#                 warnings.warn(
#                     f"{func_or_class.__name__} is deprecated and will be removed in future versions: {reason}",
#                     category=DeprecationWarning,
#                     stacklevel=2,
#                 )
#                 original_init(self, *args, **kwargs)
#
#             func_or_class.__init__ = new_init
#             return func_or_class
#
#         # Handle functions
#         @wraps(func_or_class)
#         def wrapped(*args, **kwargs):
#             warnings.warn(
#                 f"{func_or_class.__name__} is deprecated: {reason}",
#                 category=DeprecationWarning,
#                 stacklevel=2,
#             )
#             return func_or_class(*args, **kwargs)
#
#         return wrapped
#
#     return decorator
