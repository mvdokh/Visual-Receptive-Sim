# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""
Layer updates in Cython: sigmoid LN and temporal RC.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport expf

cnp.import_array()


def sigmoid_ln(
    cnp.ndarray[cnp.float32_t, ndim=2] x not None,
    float r_max,
    float x_half,
    float slope,
) -> cnp.ndarray:
    """
    r_max / (1 + exp(-slope * (x - x_half))) elementwise.
    """
    cdef cnp.npy_intp h = x.shape[0]
    cdef cnp.npy_intp w = x.shape[1]
    cdef cnp.ndarray[cnp.float32_t, ndim=2] out = np.empty_like(x)

    cdef cnp.float32_t[:, :] xin = x
    cdef cnp.float32_t[:, :] xout = out
    cdef cnp.npy_intp i, j
    cdef float val, tmp

    for i in range(h):
        for j in range(w):
            val = xin[i, j]
            tmp = slope * (val - x_half)
            # r_max / (1 + exp(-tmp))
            xout[i, j] = <cnp.float32_t>(r_max / (1.0 + expf(-tmp)))

    return out


def temporal_rc(
    cnp.ndarray[cnp.float32_t, ndim=2] state not None,
    cnp.ndarray[cnp.float32_t, ndim=2] target not None,
    float alpha,
) -> None:
    """
    In-place RC update:
        state += alpha * (target - state)
    Caller must clamp alpha to [0, 1].
    """
    cdef cnp.npy_intp h = state.shape[0]
    cdef cnp.npy_intp w = state.shape[1]

    cdef cnp.float32_t[:, :] s = state
    cdef cnp.float32_t[:, :] t = target
    cdef cnp.npy_intp i, j
    cdef float s_ij, t_ij

    for i in range(h):
        for j in range(w):
            s_ij = s[i, j]
            t_ij = t[i, j]
            s[i, j] = <cnp.float32_t>(s_ij + alpha * (t_ij - s_ij))

    return None

