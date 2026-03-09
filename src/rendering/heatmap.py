from __future__ import annotations

"""
Utilities to convert NumPy activation grids into RGBA textures
for consumption by ModernGL and Dear PyGui.
Block-average downsampling for large-field display at screen resolution.
"""

from typing import Literal, Tuple

import numpy as np


def block_average_downsample(grid: np.ndarray, max_side: int = 1024) -> np.ndarray:
    """
    Downsample grid to at most max_side per dimension using block-average.
    Never use nearest-neighbor (biologically meaningless aliasing).
    """
    h, w = grid.shape
    if h <= max_side and w <= max_side:
        return grid
    factor_h = max(1, (h + max_side - 1) // max_side)
    factor_w = max(1, (w + max_side - 1) // max_side)
    # Pad to multiple
    h2 = (h // factor_h) * factor_h
    w2 = (w // factor_w) * factor_w
    g = grid[:h2, :w2].astype(np.float32)
    new_h, new_w = h2 // factor_h, w2 // factor_w
    out = g.reshape(new_h, factor_h, new_w, factor_w).mean(axis=(1, 3))
    return out.astype(np.float32)


def block_average_downsample_rgba(rgba: np.ndarray, max_side: int = 1024) -> np.ndarray:
    """Downsample (H, W, 4) RGBA to at most max_side per dimension using block-average."""
    h, w = rgba.shape[0], rgba.shape[1]
    if h <= max_side and w <= max_side:
        return rgba
    factor_h = max(1, (h + max_side - 1) // max_side)
    factor_w = max(1, (w + max_side - 1) // max_side)
    h2, w2 = (h // factor_h) * factor_h, (w // factor_w) * factor_w
    g = rgba[:h2, :w2]
    new_h, new_w = h2 // factor_h, w2 // factor_w
    out = g.reshape(new_h, factor_h, new_w, factor_w, 4).mean(axis=(1, 3))
    return out.astype(np.float32)

ColormapName = Literal["biphasic", "firing", "spectral", "diverging"]


def _wavelength_to_rgb_vec(w: np.ndarray, gamma: float = 0.8) -> np.ndarray:
    """Vectorized wavelength (380-750 nm) to RGB (0-1). Dan Bruton's algorithm."""
    w = np.clip(w, 380.0, 750.0)
    r = np.zeros_like(w)
    g = np.zeros_like(w)
    b = np.zeros_like(w)
    m = w < 440
    atten = np.where(m, 0.3 + 0.7 * (w - 380) / 60, 1.0)
    r = np.where(m, (-(w - 440) / 60) * atten, r)
    b = np.where(m, 1.0 * atten, b)
    m = (w >= 440) & (w < 490)
    g = np.where(m, (w - 440) / 50, g)
    b = np.where(m, 1.0, b)
    m = (w >= 490) & (w < 510)
    g = np.where(m, 1.0, g)
    b = np.where(m, -(w - 510) / 20, b)
    m = (w >= 510) & (w < 580)
    r = np.where(m, (w - 510) / 70, r)
    g = np.where(m, 1.0, g)
    m = (w >= 580) & (w < 645)
    r = np.where(m, 1.0, r)
    g = np.where(m, -(w - 645) / 65, g)
    m = w >= 645
    atten = np.where(m, 0.3 + 0.7 * (750 - w) / 105, 1.0)
    r = np.where(m, 1.0 * atten, r)
    return np.stack([
        np.clip(r ** gamma, 0.0, 1.0),
        np.clip(g ** gamma, 0.0, 1.0),
        np.clip(b ** gamma, 0.0, 1.0),
    ], axis=-1)


def spectrum_to_stimulus_rgba(
    spectrum: np.ndarray, wavelengths: np.ndarray, intensity_scale: float = 0.25
) -> np.ndarray:
    """
    Convert stimulus spectrum (H, W, L) to RGBA for display.
    Uses weighted-centroid wavelength for hue, total power for brightness.
    intensity_scale: maps raw power to 0-1 (tuned so intensity≈1 appears near full).
    """
    total = np.sum(spectrum, axis=-1)
    total = np.maximum(total, 1e-9)
    centroid = np.sum(spectrum * wavelengths.astype(np.float32)[None, None, :], axis=-1) / total
    rgb3 = _wavelength_to_rgb_vec(centroid)
    bright = np.minimum(1.0, total * intensity_scale)
    rgba = np.zeros((*spectrum.shape[:2], 4), dtype=np.float32)
    rgba[..., :3] = rgb3 * bright[..., None]
    rgba[..., 3] = bright
    return rgba


def _normalize(grid: np.ndarray) -> np.ndarray:
    g = grid.astype(np.float32)
    g_min = float(np.nanmin(g))
    g_max = float(np.nanmax(g))
    if g_max <= g_min + 1e-6:
        return np.zeros_like(g)
    return (g - g_min) / (g_max - g_min)


def grid_to_rgba(
    grid: np.ndarray,
    colormap: ColormapName = "firing",
    biphasic_center: float = 0.0,
) -> np.ndarray:
    """
    Map a 2D activation grid (H, W) to an RGBA array (H, W, 4), float32 0–1.
    """
    g = grid.astype(np.float32)
    h, w = g.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)

    if colormap == "biphasic":
        # Blue (-) → black (0) → red (+)
        neg = np.clip(-(g - biphasic_center), 0.0, None)
        pos = np.clip((g - biphasic_center), 0.0, None)
        neg = _normalize(neg)
        pos = _normalize(pos)
        if not (neg.any() or pos.any()) and float(np.mean(np.abs(g))) > 0.0:
            # Uniform non-zero field: show as mild activation.
            pos[...] = 0.7
        rgba[..., 0] = pos  # R
        rgba[..., 2] = neg  # B
        rgba[..., 3] = np.maximum(neg, pos)
    elif colormap == "firing":
        # Black → amber → white
        # For firing, treat uniform non-zero fields as visible.
        if float(np.max(g)) > 0.0:
            n = g / float(np.max(g))
        else:
            n = _normalize(g)
        if not n.any() and float(np.mean(g)) > 0.0:
            n[...] = 0.7
        rgba[..., 0] = np.minimum(1.0, n * 2.0)  # R
        rgba[..., 1] = np.minimum(1.0, n * 1.2)  # G
        rgba[..., 2] = np.minimum(1.0, n * 0.5)  # B
        rgba[..., 3] = n
    elif colormap == "spectral":
        # Simple violet→red gradient over [0,1]
        n = _normalize(g)
        rgba[..., 0] = n  # R
        rgba[..., 1] = np.sin(n * np.pi)  # G-ish band
        rgba[..., 2] = 1.0 - n  # B
        rgba[..., 3] = np.clip(n * 1.2, 0.0, 1.0)
    elif colormap == "diverging":
        # Green (-) → white (0) → magenta (+)
        neg = np.clip(-g, 0.0, None)
        pos = np.clip(g, 0.0, None)
        neg = _normalize(neg)
        pos = _normalize(pos)
        rgba[..., 0] = pos
        rgba[..., 1] = np.maximum(neg, pos)
        rgba[..., 2] = neg
        rgba[..., 3] = np.maximum(neg, pos)
    else:
        raise ValueError(f"Unknown colormap: {colormap}")

    return rgba

