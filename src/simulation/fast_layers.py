from __future__ import annotations

import numpy as np


def sigmoid_ln(x: np.ndarray, r_max: float, x_half: float, slope: float) -> np.ndarray:
    """
    Fast sigmoid LN via Cython when available, else NumPy fallback.
    """
    x_f32 = np.asarray(x, dtype=np.float32)
    try:
        from hot_numerical.layer_update import sigmoid_ln as _fast

        return _fast(x_f32, float(r_max), float(x_half), float(slope))
    except ImportError:
        return r_max / (1.0 + np.exp(-slope * (x_f32 - x_half)))


def temporal_rc(state_arr: np.ndarray, target_arr: np.ndarray, alpha: float) -> None:
    """
    In-place temporal RC update, using Cython when available.

    Parameters
    ----------
    state_arr : np.ndarray
        Smoothed state array (updated in-place).
    target_arr : np.ndarray
        Raw target array.
    alpha : float
        Pre-clamped smoothing factor in [0, 1].
    """
    state_f32 = np.asarray(state_arr, dtype=np.float32)
    target_f32 = np.asarray(target_arr, dtype=np.float32)
    try:
        from hot_numerical.layer_update import temporal_rc as _fast

        _fast(state_f32, target_f32, float(alpha))
    except ImportError:
        state_f32[:] = state_f32 + alpha * (target_f32 - state_f32)

