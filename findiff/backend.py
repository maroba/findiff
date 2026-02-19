"""Backend dispatch for array operations.

Provides helpers to support numpy, JAX, and CuPy arrays transparently
in the hot path (operator application).  Construction-time and matrix-path
code remains numpy/scipy only.
"""

import numpy as np


def get_namespace(*arrays):
    """Return the array module for the given arrays.

    Checks each array in order and returns the first non-numpy namespace
    found.  Falls back to numpy if all arrays are numpy or none are given.
    """
    for arr in arrays:
        module = type(arr).__module__.split(".")[0]
        if module in ("jax", "jaxlib"):
            import jax.numpy as jnp

            return jnp
        if module == "cupy":
            import cupy

            return cupy
    return np


def is_array(obj):
    """Check whether *obj* is an array (numpy, JAX, CuPy, etc.).

    Uses duck typing: any object with ``shape`` and ``dtype`` attributes
    is considered an array.
    """
    return hasattr(obj, "shape") and hasattr(obj, "dtype")


def add_at(arr, idx, val):
    """Perform ``arr[idx] += val`` for any backend.

    JAX arrays are immutable so in-place addition is not possible.
    This helper uses ``arr.at[idx].add(val)`` for JAX and falls back to
    standard in-place ``+=`` for numpy/CuPy.

    Always returns the (possibly new) array.
    """
    module = type(arr).__module__.split(".")[0]
    if module in ("jax", "jaxlib"):
        return arr.at[idx].add(val)
    arr[idx] += val
    return arr
