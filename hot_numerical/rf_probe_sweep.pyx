# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""
RF probe sweep — bilinear interpolation on a float32 firing-rate grid.

probe_sweep(fr_grid, probe_xs, probe_ys)
  - fr_grid: (H, W) float32
  - probe_xs, probe_ys: 1D float32 arrays of fractional pixel coordinates
    (x: column index, y: row index)
  - returns (N,) float32 interpolated values
"""

import numpy as np
cimport numpy as cnp

cnp.import_array()


def probe_sweep(
    cnp.ndarray[cnp.float32_t, ndim=2] fr_grid not None,
    cnp.ndarray[cnp.float32_t, ndim=1] probe_xs not None,
    cnp.ndarray[cnp.float32_t, ndim=1] probe_ys not None,
) -> cnp.ndarray:
    """
    Bilinear interpolation of fr_grid at positions (probe_xs[k], probe_ys[k]).
    Coordinates are clamped to the grid edges.
    """
    cdef cnp.npy_intp h = fr_grid.shape[0]
    cdef cnp.npy_intp w = fr_grid.shape[1]
    cdef cnp.npy_intp n = probe_xs.shape[0]

    cdef cnp.ndarray[cnp.float32_t, ndim=1] out = np.empty(n, dtype=np.float32)

    cdef cnp.float32_t[:, :] grid = fr_grid
    cdef cnp.float32_t[:] xs = probe_xs
    cdef cnp.float32_t[:] ys = probe_ys
    cdef cnp.float32_t[:] o = out

    cdef cnp.npy_intp k
    cdef float x, y, dx, dy
    cdef cnp.npy_intp x0, y0, x1, y1
    cdef float v00, v01, v10, v11

    if n != probe_ys.shape[0]:
        raise ValueError("probe_xs and probe_ys must have the same length")

    for k in range(n):
        x = xs[k]
        y = ys[k]

        # Clamp to [0, w-1] and [0, h-1]
        if x < 0.0:
            x = 0.0
        elif x > w - 1:
            x = w - 1
        if y < 0.0:
            y = 0.0
        elif y > h - 1:
            y = h - 1

        x0 = <cnp.npy_intp>x
        y0 = <cnp.npy_intp>y
        x1 = x0 + 1
        y1 = y0 + 1
        if x1 >= w:
            x1 = w - 1
        if y1 >= h:
            y1 = h - 1

        dx = x - x0
        dy = y - y0

        v00 = grid[y0, x0]
        v01 = grid[y0, x1]
        v10 = grid[y1, x0]
        v11 = grid[y1, x1]

        o[k] = <cnp.float32_t>(
            (1.0 - dx) * (1.0 - dy) * v00
            + dx * (1.0 - dy) * v01
            + (1.0 - dx) * dy * v10
            + dx * dy * v11
        )

    return out

