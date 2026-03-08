# cython: language_level=3
"""
Template: RF probe sweep — evaluate RGC response at many probe positions.
"""
import numpy as np
cimport numpy as cnp

cnp.import_array()


def probe_sweep(
    cnp.ndarray[cnp.float32_t, ndim=2] fr_grid not None,
    cnp.ndarray[cnp.float32_t, ndim=1] x_deg not None,
    cnp.ndarray[cnp.float32_t, ndim=1] y_deg not None,
) -> cnp.ndarray:
    """
    Placeholder: for each (x,y) probe position, sample fr_grid.
    Bilinear interpolation or nearest-neighbor.
    """
    cdef cnp.npy_intp n = x_deg.shape[0]
    cdef cnp.ndarray[cnp.float32_t, ndim=2] rf_map = np.zeros((n, n), dtype=np.float32)
    # TODO: implement interpolation over grid
    rf_map[:] = 0.0
    return rf_map
