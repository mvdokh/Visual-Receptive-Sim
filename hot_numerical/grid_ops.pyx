# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Fast reductions and (H,W,L)·l -> (H,W) for the retinal pipeline and UI probes.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport INFINITY, isnan

cnp.import_array()


def mean_f32_2d(cnp.ndarray[cnp.float32_t, ndim=2] arr not None) -> float:
    """Mean of all elements; requires C-contiguous float32."""
    cdef cnp.npy_intp h = arr.shape[0]
    cdef cnp.npy_intp w = arr.shape[1]
    cdef cnp.npy_intp i, j
    cdef cnp.float32_t[:, :] a = arr
    cdef double s = 0.0
    cdef cnp.npy_intp n = h * w
    if n == 0:
        return 0.0
    for i in range(h):
        for j in range(w):
            s += <double>a[i, j]
    return <float>(s / <double>n)


def spectral_dot_hwl_l_to_hw(
    cnp.ndarray[cnp.float32_t, ndim=3] stim not None,
    cnp.ndarray[cnp.float32_t, ndim=1] sens not None,
) -> cnp.ndarray:
    """
    For stim (H, W, L) and sensitivity vector (L,), return (H, W) float32
    dot product at each pixel: out[i,j] = sum_k stim[i,j,k] * sens[k].
    """
    cdef cnp.npy_intp h = stim.shape[0]
    cdef cnp.npy_intp w = stim.shape[1]
    cdef cnp.npy_intp l = stim.shape[2]
    if sens.shape[0] != l:
        raise ValueError("stim depth must match sens length")

    cdef cnp.ndarray[cnp.float32_t, ndim=2] out = np.empty((h, w), dtype=np.float32)
    cdef cnp.float32_t[:, :, :] st = stim
    cdef cnp.float32_t[:] se = sens
    cdef cnp.float32_t[:, :] ou = out
    cdef cnp.npy_intp i, j, k
    cdef double acc

    for i in range(h):
        for j in range(w):
            acc = 0.0
            for k in range(l):
                acc += <double>st[i, j, k] * <double>se[k]
            ou[i, j] = <cnp.float32_t>acc
    return out


def fr_histogram_16bins_subsampled(
    cnp.ndarray[cnp.float32_t, ndim=2] arr not None,
    cnp.npy_intp max_samples,
) -> tuple:
    """
    Strided subsample (at most max_samples values), then 16 uniform bins.
    Returns (counts_f64 length 16, edge0, edge15_end) for bar-plot centers in Python.
    Ignores non-finite values.
    """
    cdef cnp.npy_intp h = arr.shape[0]
    cdef cnp.npy_intp w = arr.shape[1]
    cdef cnp.npy_intp total = h * w
    cdef cnp.npy_intp stride = 1
    if total > max_samples and max_samples > 0:
        stride = total // max_samples
        if stride < 1:
            stride = 1

    cdef cnp.float32_t[:, :] a = arr
    cdef cnp.npy_intp ii, jj, t = 0
    cdef float v
    cdef float mn = <float>INFINITY
    cdef float mx = -<float>INFINITY
    cdef cnp.npy_intp seen = 0

    for ii in range(h):
        for jj in range(w):
            if t % stride != 0:
                t += 1
                continue
            t += 1
            v = a[ii, jj]
            if isnan(<double>v):
                continue
            seen += 1
            if v < mn:
                mn = v
            if v > mx:
                mx = v

    if seen == 0:
        return np.zeros(16, dtype=np.float64), 0.0, 1.0

    if mx <= mn:
        mx = mn + <float>1.0

    cdef double span = <double>mx - <double>mn
    cdef double inv = 16.0 / span
    cdef cnp.npy_intp b
    cdef cnp.ndarray[cnp.float64_t, ndim=1] counts = np.zeros(16, dtype=np.float64)
    cdef double[:] cnt = counts

    t = 0
    for ii in range(h):
        for jj in range(w):
            if t % stride != 0:
                t += 1
                continue
            t += 1
            v = a[ii, jj]
            if isnan(<double>v):
                continue
            b = <cnp.npy_intp>((<double>v - <double>mn) * inv)
            if b < 0:
                b = 0
            elif b > 15:
                b = 15
            cnt[b] += 1.0

    cdef float edge0 = mn
    cdef float edge_end = mx
    return counts, edge0, edge_end
