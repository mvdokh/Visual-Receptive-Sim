from __future__ import annotations

"""
Utilities to convert NumPy activation grids into RGBA textures
for consumption by ModernGL and Dear PyGui.
"""

from typing import Literal

import numpy as np

ColormapName = Literal["biphasic", "firing", "spectral", "diverging"]


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

