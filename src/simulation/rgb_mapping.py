from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


_RGB_WAVELENGTHS: np.ndarray | None = None
_RGB_PRIMARIES: np.ndarray | None = None  # shape (3, Lrgb)
_GAMMA_LUT: np.ndarray | None = None      # shape (256,)


def _ensure_rgb_tables() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load RGB spectral primaries and gamma LUT (cached)."""
    global _RGB_WAVELENGTHS, _RGB_PRIMARIES, _GAMMA_LUT
    if _RGB_WAVELENGTHS is None or _RGB_PRIMARIES is None:
        root = Path(__file__).resolve().parents[2]
        rgb_path = root / "rgbtolms" / "RGB.txt"
        data = np.loadtxt(rgb_path, dtype=np.float32)
        wl = data[:, 0].astype(np.float32)
        spd = data[:, 1:].astype(np.float32).T  # (3, Lrgb)
        _RGB_WAVELENGTHS = wl
        _RGB_PRIMARIES = spd
    if _GAMMA_LUT is None:
        # Approximate the original rgbtolms gamma_fcn (x ** 2.2 on 0-1 grid).
        x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        _GAMMA_LUT = x ** 2.2
    return _RGB_WAVELENGTHS, _RGB_PRIMARIES, _GAMMA_LUT


def build_emission_from_rgb(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map gamma-encoded RGB in [0,1] to a spectral power distribution using monitor primaries.

    Args:
        rgb: array (..., 3) float32 in [0,1].

    Returns:
        wavelengths_rgb: (Lrgb,) float32 wavelength grid.
        emission_spd: (..., Lrgb) float32, same leading shape as rgb.
    """
    wl, primaries, gamma_lut = _ensure_rgb_tables()
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.shape[-1] != 3:
        raise ValueError(f"RGB array must have last dimension 3, got shape {arr.shape}")
    arr = np.clip(arr, 0.0, 1.0)
    idx = (arr * 255.0).astype(np.int32)
    idx = np.clip(idx, 0, 255)
    gamma_rgb = gamma_lut[idx]  # (..., 3)
    flat = gamma_rgb.reshape(-1, 3)  # (N, 3)
    # primaries: (3, Lrgb) -> emission_flat: (N, Lrgb)
    emission_flat = flat @ primaries
    emission = emission_flat.reshape(arr.shape[:-1] + (primaries.shape[1],))
    return wl, emission


def resample_spd_to_spectral_grid(
    wavelengths_src: np.ndarray,
    spd_src: np.ndarray,
    wavelengths_dst: np.ndarray,
) -> np.ndarray:
    """
    Resample source SPD to the simulator's spectral wavelength grid using linear interpolation.

    Shapes:
        wavelengths_src: (Lsrc,)
        spd_src: (..., Lsrc)
        wavelengths_dst: (Ldst,)
    Returns:
        spd_dst: (..., Ldst)
    """
    wl_src = np.asarray(wavelengths_src, dtype=np.float32)
    wl_dst = np.asarray(wavelengths_dst, dtype=np.float32)
    spd = np.asarray(spd_src, dtype=np.float32)
    flat = spd.reshape(-1, spd.shape[-1])  # (N, Lsrc)
    out_flat = np.empty((flat.shape[0], wl_dst.shape[0]), dtype=np.float32)
    for i in range(flat.shape[0]):
        out_flat[i] = np.interp(wl_dst, wl_src, flat[i])
    return out_flat.reshape(spd.shape[:-1] + (wl_dst.shape[0],))

