# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""
Fast stimulus fill for spot and full_field without creating full meshgrids.
Fills spectrum (h, w, nwl) in a single pass.
"""

import numpy as np
cimport numpy as cnp
cnp.import_array()


def fill_spot_or_full(
    cnp.npy_intp h,
    cnp.npy_intp w,
    float dx_deg,
    float cx_deg,
    float cy_deg,
    float radius_deg,
    cnp.ndarray[cnp.float32_t, ndim=1] profile not None,
    cnp.ndarray[cnp.float32_t, ndim=3] out not None,
    bint full_field,
) -> None:
    """
    Fill out (h, w, nwl) with stimulus.
    If full_field: all ones * profile.
    Else: circular spot mask (r <= radius_deg) * profile.
    Center (cx_deg, cy_deg); grid spacing dx_deg; origin at grid center.
    """
    cdef cnp.npy_intp nwl = profile.shape[0]
    cdef cnp.npy_intp i, j, k
    cdef float x_deg, y_deg, r, mask_val
    cdef float r2 = radius_deg * radius_deg
    # Grid center in pixels (same as Python: arange - size/2 + 0.5 => center at (size-1)/2)
    cdef float center_x = (w - 1) * 0.5
    cdef float center_y = (h - 1) * 0.5

    cdef cnp.float32_t[:] p = profile
    cdef cnp.float32_t[:, :, :] o = out

    for i in range(h):
        # y_deg at pixel i: (i - h/2 + 0.5)*dx_deg; then offset from spot center cy_deg
        y_deg = (i - center_y) * dx_deg - cy_deg
        for j in range(w):
            x_deg = (j - center_x) * dx_deg - cx_deg
            if full_field:
                mask_val = 1.0
            else:
                r = x_deg * x_deg + y_deg * y_deg
                mask_val = 1.0 if r <= r2 else 0.0
            for k in range(nwl):
                o[i, j, k] = mask_val * p[k]
    return None
