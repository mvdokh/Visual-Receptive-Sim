from __future__ import annotations

import os
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

# σ at or below this (pixels) uses Cython separable blur when built; larger σ uses
# FFT Gaussian via ``smart_gaussian`` (often faster than ``gaussian_filter`` here).
SIGMA_FFT_THRESHOLD_PX = 15.0

# Load Cython implementation once at import so hot_numerical is found when run from repo root
_cython_gaussian_pool_2d = None
try:
    from hot_numerical.convolve_2d import gaussian_pool_2d as _cython_gaussian_pool_2d
except ImportError:
    pass


def _fft_gaussian_2d(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur via FFT (no mode; assumes reflect-like by padding)."""
    arr = np.asarray(arr, dtype=np.float32)
    h, w = arr.shape
    # Kernel size: odd, at least 6*sigma each side
    k = max(3, int(6 * sigma + 1)) | 1
    half = k // 2
    y = np.arange(-half, half + 1, dtype=np.float32)
    x = np.arange(-half, half + 1, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    kernel = np.exp(-0.5 * (xx**2 + yy**2) / (sigma**2)).astype(np.float32)
    kernel /= kernel.sum()
    # Pad array to reduce boundary artifacts (reflect)
    pad = half
    padded = np.pad(arr, ((pad, pad), (pad, pad)), mode="reflect")
    out = fftconvolve(padded, kernel, mode="same")
    return out[pad : pad + h, pad : pad + w].astype(np.float32)


def smart_gaussian(
    array: np.ndarray, sigma_px: float, mode: str = "reflect"
) -> np.ndarray:
    """
    Gaussian blur: use FFT for large sigma (faster), direct filter for small sigma.
    For large-field grids (e.g. 2048×2048), sigma > 15 px should use FFT.
    """
    arr = np.asarray(array, dtype=np.float32)
    if sigma_px <= SIGMA_FFT_THRESHOLD_PX:
        return gaussian_filter(arr, sigma_px, mode=mode)
    return _fft_gaussian_2d(arr, sigma_px)


def gaussian_pool_2d(
    arr: np.ndarray, sigma: float, mode: str = "reflect", use_smart: bool = True
) -> np.ndarray:
    """
    Prefer Cython for small σ; for larger σ use FFT convolution (``smart_gaussian``).

    ``HOT_NUMERICAL_USE_CYTHON_CONV=0`` skips Cython. ``use_smart=False`` skips the FFT
    path and uses ``gaussian_filter`` for all σ.
    """
    arr_f32 = np.asarray(arr, dtype=np.float32)

    use_cython = os.environ.get("HOT_NUMERICAL_USE_CYTHON_CONV", "1") != "0"
    if (
        use_cython
        and sigma <= SIGMA_FFT_THRESHOLD_PX
        and _cython_gaussian_pool_2d is not None
    ):
        return _cython_gaussian_pool_2d(arr_f32, float(sigma), mode)

    if use_smart and sigma > SIGMA_FFT_THRESHOLD_PX:
        return smart_gaussian(arr_f32, sigma, mode=mode)
    return gaussian_filter(arr_f32, float(sigma), mode=mode)


