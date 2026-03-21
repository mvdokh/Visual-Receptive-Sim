from __future__ import annotations

"""
Utilities to convert NumPy activation grids into RGBA textures
for consumption by ModernGL and Dear PyGui.
Block-average downsampling for large-field display at screen resolution.
"""

from typing import Literal, Optional, Tuple

import numpy as np

# Scale bar: 100 µm default (bio_constants.SCALE_BAR_UM)
# Literature: Masland 2012 (200 µm total thickness), Curcio et al. 1992 (cone diameter)
DEFAULT_SCALE_BAR_UM = 100.0


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


def _intensity_to_firing01(g: np.ndarray) -> np.ndarray:
    """
    Same intensity scale as the \"firing\" (amber) heatmap: n = g / max(g) when
    max > 0, else min-max normalize, plus the uniform-nonzero fallback.

    All colormaps use this n so spectral / diverging / biphasic show the same
    underlying activity as amber; only the RGB mapping differs.
    """
    g = np.asarray(g, dtype=np.float32)
    if float(np.max(g)) > 0.0:
        n = g / float(np.max(g))
    else:
        n = _normalize(g)
    if not n.any() and float(np.mean(g)) > 0.0:
        n = np.full_like(g, 0.7, dtype=np.float32)
    return n.astype(np.float32)


def _signed_from_intensity01(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map n in [0, 1] (amber intensity) to symmetric signed [-1, 1] for diverging/biphasic hues.
    """
    n_clip = np.clip(n, 0.0, 1.0)
    g_signed = 2.0 * n_clip - 1.0
    neg = np.clip(-g_signed, 0.0, None)
    pos = np.clip(g_signed, 0.0, None)
    return neg, pos


def grid_to_rgba(
    grid: np.ndarray,
    colormap: ColormapName = "firing",
    biphasic_center: float = 0.0,
) -> np.ndarray:
    """
    Map a 2D activation grid (H, W) to an RGBA array (H, W, 4), float32 0-1.

    Every colormap uses the same intensity normalization as \"firing\" (g to n);
    only the color mapping from n differs.
    """
    g = grid.astype(np.float32)
    h, w = g.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)

    # Single intensity pipeline: match amber (optionally offset for biphasic center)
    if colormap == "biphasic":
        n = _intensity_to_firing01(g - float(biphasic_center))
    else:
        n = _intensity_to_firing01(g)

    if colormap == "firing":
        rgba[..., 0] = np.minimum(1.0, n * 2.0)  # R
        rgba[..., 1] = np.minimum(1.0, n * 1.2)  # G
        rgba[..., 2] = np.minimum(1.0, n * 0.5)  # B
        rgba[..., 3] = n
    elif colormap == "spectral":
        nc = np.clip(n, 0.0, 1.0)
        rgba[..., 0] = nc
        rgba[..., 1] = np.sin(nc * np.pi)
        rgba[..., 2] = 1.0 - nc
        rgba[..., 3] = np.clip(nc * 1.2, 0.0, 1.0)
    elif colormap == "diverging":
        neg, pos = _signed_from_intensity01(n)
        rgba[..., 0] = pos
        rgba[..., 1] = np.maximum(neg, pos)
        rgba[..., 2] = neg
        rgba[..., 3] = np.maximum(neg, pos)
    elif colormap == "biphasic":
        neg, pos = _signed_from_intensity01(n)
        rgba[..., 0] = pos
        rgba[..., 2] = neg
        rgba[..., 3] = np.maximum(neg, pos)
    else:
        raise ValueError(f"Unknown colormap: {colormap}")

    return rgba


def draw_scale_bar_rgba(
    rgba: np.ndarray,
    microns_per_px: float,
    scale_bar_um: float = DEFAULT_SCALE_BAR_UM,
    position: Literal["bottom_left", "bottom_right"] = "bottom_left",
    bar_height_px: int = 4,
    margin_px: int = 12,
    color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Draw a scale bar onto the RGBA array in-place.

    Scale bar length in pixels = scale_bar_um / microns_per_px.
    References: Masland 2012 (retinal thickness), Curcio et al. 1992 (cone diameter).
    """
    h, w = rgba.shape[0], rgba.shape[1]
    bar_length_px = int(scale_bar_um / microns_per_px)
    if bar_length_px < 2 or bar_length_px > w // 2:
        return rgba
    y0 = h - margin_px - bar_height_px
    y1 = h - margin_px
    if position == "bottom_right":
        x0 = w - margin_px - bar_length_px
    else:
        x0 = margin_px
    x1 = x0 + bar_length_px
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    rgba[y0:y1, x0:x1, :] = color
    # Label (simple white bar; text would require font rendering)
    return rgba
