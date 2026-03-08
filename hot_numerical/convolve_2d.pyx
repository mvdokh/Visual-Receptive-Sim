# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""
Separable 2D Gaussian pooling for float32 grids.

gaussian_pool_2d(arr, sigma, mode="reflect")
  - arr: float32 2D array
  - sigma: standard deviation in pixels
  - mode: "reflect" or "constant" (zero outside)
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport exp

cnp.import_array()


def gaussian_pool_2d(
    cnp.ndarray[cnp.float32_t, ndim=2] arr not None,
    float sigma,
    mode="reflect",
) -> cnp.ndarray:
    """
    Apply separable 2D Gaussian pooling to a float32 array.
    """
    cdef cnp.npy_intp h = arr.shape[0]
    cdef cnp.npy_intp w = arr.shape[1]

    if sigma <= 0.0:
        # Degenerate: return a copy
        return np.asarray(arr, dtype=np.float32).copy()

    cdef float three_sigma = 3.0 * sigma
    cdef int half = <int>(three_sigma + 0.5)
    if half < 1:
        half = 1
    cdef int size = 2 * half + 1

    # Build 1D Gaussian kernel
    cdef cnp.ndarray[cnp.float32_t, ndim=1] kernel = np.empty(size, dtype=np.float32)
    cdef int i
    cdef float x, s2 = 2.0 * sigma * sigma
    cdef float sum_w = 0.0
    for i in range(size):
        x = <float>(i - half)
        kernel[i] = <cnp.float32_t>exp(-(x * x) / s2)
        sum_w += kernel[i]
    if sum_w != 0.0:
        for i in range(size):
            kernel[i] /= sum_w

    cdef cnp.ndarray[cnp.float32_t, ndim=2] tmp = np.empty((h, w), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2] out = np.empty((h, w), dtype=np.float32)

    cdef cnp.float32_t[:, :] a = arr
    cdef cnp.float32_t[:, :] t = tmp
    cdef cnp.float32_t[:, :] o = out
    cdef cnp.float32_t[:] k = kernel

    cdef int mode_reflect = 1 if mode == "reflect" else 0

    cdef cnp.npy_intp y, xj, xi
    cdef int k_idx
    cdef float acc, val
    cdef cnp.npy_intp jj, ii

    # Horizontal pass (rows)
    for y in range(h):
        for xj in range(w):
            acc = 0.0
            for k_idx in range(size):
                jj = xj + (k_idx - half)
                if 0 <= jj < w:
                    val = a[y, jj]
                else:
                    if mode_reflect:
                        if jj < 0:
                            jj = -jj - 1
                        while jj >= w:
                            jj = 2 * w - jj - 1
                        val = a[y, jj]
                    else:
                        val = 0.0
                acc += k[k_idx] * val
            t[y, xj] = <cnp.float32_t>acc

    # Vertical pass (columns)
    for xj in range(w):
        for yi in range(h):
            acc = 0.0
            for k_idx in range(size):
                ii = yi + (k_idx - half)
                if 0 <= ii < h:
                    val = t[ii, xj]
                else:
                    if mode_reflect:
                        if ii < 0:
                            ii = -ii - 1
                        while ii >= h:
                            ii = 2 * h - ii - 1
                        val = t[ii, xj]
                    else:
                        val = 0.0
                acc += k[k_idx] * val
            o[yi, xj] = <cnp.float32_t>acc

    return out

