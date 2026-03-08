from __future__ import annotations

import os
import numpy as np
from scipy.ndimage import gaussian_filter


def gaussian_pool_2d(arr: np.ndarray, sigma: float, mode: str = "reflect") -> np.ndarray:
    """
    Wrapper around a fast Cython separable Gaussian pool, with SciPy fallback.

    By default we use SciPy's highly optimized ``gaussian_filter`` which is
    very fast and well-tested. If you want to force the Cython implementation,
    set the environment variable ``HOT_NUMERICAL_USE_CYTHON_CONV=1``.
    """
    arr_f32 = np.asarray(arr, dtype=np.float32)

    use_cython = os.environ.get("HOT_NUMERICAL_USE_CYTHON_CONV") == "1"
    if use_cython:
        try:
            from hot_numerical.convolve_2d import gaussian_pool_2d as _fast

            return _fast(arr_f32, float(sigma), mode)
        except ImportError:
            # Fall back to SciPy if the extension is not available
            pass

    return gaussian_filter(arr_f32, sigma, mode=mode)


