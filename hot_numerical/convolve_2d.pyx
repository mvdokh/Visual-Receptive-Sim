# cython: language_level=3
"""
Template: 2D convolution / Gaussian pooling for horizontal cell surround, etc.
"""
import numpy as np
cimport numpy as cnp

cnp.import_array()


def gaussian_pool_2d(cnp.ndarray[cnp.float32_t, ndim=2] grid not None,
                    float sigma) -> cnp.ndarray:
    """
    Placeholder: apply 2D Gaussian pooling over grid.
    Replace with scipy.ndimage.gaussian_filter or hand-rolled separable kernel.
    """
    cdef cnp.npy_intp h = grid.shape[0]
    cdef cnp.npy_intp w = grid.shape[1]
    cdef cnp.ndarray[cnp.float32_t, ndim=2] out = np.empty((h, w), dtype=np.float32)
    # TODO: implement separable Gaussian convolution
    out[:] = grid
    return out
