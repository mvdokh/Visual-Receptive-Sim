# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Separable 2D Gaussian pooling for float32 grids. Optimized with nogil and raw pointers.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport exp

cnp.import_array()

cdef inline int _reflect(int j, int n) nogil:
    """Reflect index j into [0, n-1]."""
    if j < 0:
        j = -j - 1
    while j >= n:
        j = 2 * n - j - 1
    return j


cdef void _horizontal_pass(
    const cnp.float32_t *a,
    cnp.float32_t *t,
    const cnp.float32_t *k,
    cnp.npy_intp h, cnp.npy_intp w,
    int half, int size, int reflect,
) nogil noexcept:
    cdef cnp.npy_intp y, xj, k_idx, jj
    cdef float acc, val
    cdef cnp.npy_intp row_off
    cdef cnp.npy_intp x_start = half
    cdef cnp.npy_intp x_end = w - half
    for y in range(h):
        row_off = y * w
        # Left boundary
        for xj in range(x_start):
            acc = 0.0
            for k_idx in range(size):
                jj = xj + (k_idx - half)
                if 0 <= jj < w:
                    val = a[row_off + jj]
                else:
                    if reflect:
                        jj = _reflect(jj, w)
                        val = a[row_off + jj]
                    else:
                        val = 0.0
                acc += k[k_idx] * val
            t[row_off + xj] = <cnp.float32_t>acc
        # Interior (no bounds check in inner loop)
        for xj in range(x_start, x_end):
            acc = 0.0
            for k_idx in range(size):
                acc += k[k_idx] * a[row_off + xj + (k_idx - half)]
            t[row_off + xj] = <cnp.float32_t>acc
        # Right boundary
        for xj in range(x_end, w):
            acc = 0.0
            for k_idx in range(size):
                jj = xj + (k_idx - half)
                if 0 <= jj < w:
                    val = a[row_off + jj]
                else:
                    if reflect:
                        jj = _reflect(jj, w)
                        val = a[row_off + jj]
                    else:
                        val = 0.0
                acc += k[k_idx] * val
            t[row_off + xj] = <cnp.float32_t>acc


cdef void _vertical_pass(
    const cnp.float32_t *t,
    cnp.float32_t *o,
    const cnp.float32_t *k,
    cnp.npy_intp h, cnp.npy_intp w,
    int half, int size, int reflect,
) nogil noexcept:
    cdef cnp.npy_intp xj, yi, k_idx, ii
    cdef float acc, val
    cdef cnp.npy_intp y_start = half
    cdef cnp.npy_intp y_end = h - half
    for xj in range(w):
        # Top boundary
        for yi in range(y_start):
            acc = 0.0
            for k_idx in range(size):
                ii = yi + (k_idx - half)
                if 0 <= ii < h:
                    val = t[ii * w + xj]
                else:
                    if reflect:
                        ii = _reflect(ii, h)
                        val = t[ii * w + xj]
                    else:
                        val = 0.0
                acc += k[k_idx] * val
            o[yi * w + xj] = <cnp.float32_t>acc
        # Interior (no bounds check in inner loop)
        for yi in range(y_start, y_end):
            acc = 0.0
            for k_idx in range(size):
                acc += k[k_idx] * t[(yi + (k_idx - half)) * w + xj]
            o[yi * w + xj] = <cnp.float32_t>acc
        # Bottom boundary
        for yi in range(y_end, h):
            acc = 0.0
            for k_idx in range(size):
                ii = yi + (k_idx - half)
                if 0 <= ii < h:
                    val = t[ii * w + xj]
                else:
                    if reflect:
                        ii = _reflect(ii, h)
                        val = t[ii * w + xj]
                    else:
                        val = 0.0
                acc += k[k_idx] * val
            o[yi * w + xj] = <cnp.float32_t>acc


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
        return np.asarray(arr, dtype=np.float32).copy()

    cdef float three_sigma = 3.0 * sigma
    cdef int half = <int>(three_sigma + 0.5)
    if half < 1:
        half = 1
    cdef int size = 2 * half + 1
    cdef int reflect = 1 if mode == "reflect" else 0

    # Build 1D kernel (Python/nogil boundary)
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

    # Contiguous buffers
    cdef cnp.ndarray[cnp.float32_t, ndim=2] tmp = np.empty((h, w), dtype=np.float32, order='C')
    cdef cnp.ndarray[cnp.float32_t, ndim=2] out = np.empty((h, w), dtype=np.float32, order='C')
    # ascontiguousarray returns input if already C-contiguous (no copy)
    cdef cnp.ndarray[cnp.float32_t, ndim=2] arr_c = np.ascontiguousarray(arr, dtype=np.float32)
    cdef cnp.float32_t[:, :] a_v = arr_c
    cdef cnp.float32_t[:, :] t_v = tmp
    cdef cnp.float32_t[:, :] o_v = out
    cdef cnp.float32_t[:] k_v = kernel
    cdef const cnp.float32_t *a_ptr = &a_v[0, 0]
    cdef cnp.float32_t *t_ptr = &t_v[0, 0]
    cdef cnp.float32_t *o_ptr = &o_v[0, 0]
    cdef const cnp.float32_t *k_ptr = &k_v[0]

    with nogil:
        _horizontal_pass(a_ptr, t_ptr, k_ptr, h, w, half, size, reflect)
        _vertical_pass(t_ptr, o_ptr, k_ptr, h, w, half, size, reflect)

    return out
