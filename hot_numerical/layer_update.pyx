# cython: language_level=3
"""
Template: single layer update — e.g. bipolar LN, amacrine lateral, RGC sigmoid.
"""
import numpy as np
cimport numpy as cnp

cnp.import_array()


def sigmoid_ln(
    cnp.ndarray[cnp.float32_t, ndim=2] x not None,
    float r_max,
    float x_half,
    float slope,
) -> cnp.ndarray:
    """
    Placeholder: r_max / (1 + exp(-slope * (x - x_half))).
    """
    cdef cnp.ndarray[cnp.float32_t, ndim=2] out = np.empty_like(x)
    # TODO: vectorized sigmoid
    out[:] = x
    return out


def temporal_rc(
    cnp.ndarray[cnp.float32_t, ndim=2] state not None,
    cnp.ndarray[cnp.float32_t, ndim=2] target not None,
    float tau,
    float dt,
) -> cnp.ndarray:
    """
    Placeholder: exponential smoothing dstate/dt = (target - state) / tau.
    """
    cdef cnp.ndarray[cnp.float32_t, ndim=2] out = np.empty_like(state)
    # TODO: out = state + (target - state) * (1 - exp(-dt/tau))
    out[:] = state
    return out
