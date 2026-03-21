"""Tests for rendering/heatmap: wavelength-to-RGB, spectrum_to_stimulus_rgba, grid_to_rgba colormaps."""
from __future__ import annotations

import numpy as np
import pytest

from src.rendering.heatmap import (
    _wavelength_to_rgb_vec,
    spectrum_to_stimulus_rgba,
    _normalize,
    _intensity_to_firing01,
    grid_to_rgba,
)


def test_wavelength_to_rgb_vec_shape():
    """Output shape is (3, N) for N wavelengths."""
    w = np.array([400.0, 550.0, 650.0])
    rgb = _wavelength_to_rgb_vec(w)
    assert rgb.shape == (3, 3)
    np.testing.assert_array_less(-0.01, rgb)
    np.testing.assert_array_less(rgb, 1.01)


def test_wavelength_to_rgb_vec_green_mid():
    """~550 nm is green (G > R, G > B)."""
    rgb = _wavelength_to_rgb_vec(np.array([550.0]))
    # shape (1, 3) for single wavelength: [R, G, B]
    assert rgb.shape == (1, 3)
    r, g, b = float(rgb[0, 0]), float(rgb[0, 1]), float(rgb[0, 2])
    assert g >= r and g >= b


def test_spectrum_to_stimulus_rgba_shape():
    """Output is (H, W, 4) float32."""
    spec = np.zeros((8, 8, 10), dtype=np.float32)
    spec[4, 4, :] = 1.0
    wl = np.linspace(380, 700, 10, dtype=np.float32)
    rgba = spectrum_to_stimulus_rgba(spec, wl)
    assert rgba.shape == (8, 8, 4)
    assert rgba.dtype == np.float32
    assert np.all(rgba >= 0) and np.all(rgba <= 1.01)


def test_spectrum_to_stimulus_rgba_intensity_scale():
    """Larger intensity_scale increases brightness."""
    spec = np.ones((4, 4, 5), dtype=np.float32) * 0.5
    wl = np.linspace(380, 700, 5, dtype=np.float32)
    rgba_lo = spectrum_to_stimulus_rgba(spec, wl, intensity_scale=0.1)
    rgba_hi = spectrum_to_stimulus_rgba(spec, wl, intensity_scale=0.5)
    assert float(np.mean(rgba_hi[..., 3])) > float(np.mean(rgba_lo[..., 3]))


def test_normalize_constant():
    """Constant grid normalizes to zeros."""
    g = np.ones((2, 2)) * 5.0
    n = _normalize(g)
    np.testing.assert_array_almost_equal(n, 0.0)


def test_normalize_range():
    """Normalize maps [a,b] to [0,1]."""
    g = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    n = _normalize(g)
    assert float(np.min(n)) == pytest.approx(0.0)
    assert float(np.max(n)) == pytest.approx(1.0)


def test_intensity_to_firing01_matches_firing_alpha():
    """_intensity_to_firing01 is the same scale as amber alpha channel."""
    g = np.array([[0.1, 0.5], [0.2, 0.0]], dtype=np.float32)
    n = _intensity_to_firing01(g)
    rgba_f = grid_to_rgba(g, colormap="firing")
    np.testing.assert_allclose(n, rgba_f[..., 3], rtol=0, atol=1e-6)


def test_intensity_to_firing01_uniform_positive():
    """Uniform positive g -> n all ones (same as g / max(g))."""
    g = np.full((4, 4), 0.42, dtype=np.float32)
    n = _intensity_to_firing01(g)
    np.testing.assert_allclose(n, 1.0)


def test_intensity_to_firing01_all_zero():
    """All-zero g -> n all zero."""
    g = np.zeros((3, 3), dtype=np.float32)
    n = _intensity_to_firing01(g)
    np.testing.assert_allclose(n, 0.0)


def test_grid_to_rgba_firing():
    """Firing colormap: output (H,W,4), alpha follows intensity."""
    g = np.array([[0.0, 0.5], [1.0, 0.2]], dtype=np.float32)
    rgba = grid_to_rgba(g, colormap="firing")
    assert rgba.shape == (2, 2, 4)
    assert np.all(rgba >= 0) and np.all(rgba <= 1.01)


def test_grid_to_rgba_biphasic():
    """Biphasic colormap runs without error."""
    g = np.array([[-0.5, 0.0], [0.5, 0.2]], dtype=np.float32)
    rgba = grid_to_rgba(g, colormap="biphasic", biphasic_center=0.0)
    assert rgba.shape == (2, 2, 4)


def test_grid_to_rgba_spectral():
    """Spectral colormap runs without error."""
    g = np.array([[0.0, 0.5], [1.0, 0.0]], dtype=np.float32)
    rgba = grid_to_rgba(g, colormap="spectral")
    assert rgba.shape == (2, 2, 4)


def test_spectral_uses_same_intensity_as_firing():
    """Spectral R channel equals clipped firing intensity n (same pipeline as amber)."""
    g = np.array([[0.1, 0.4], [0.0, 0.8]], dtype=np.float32)
    n = _intensity_to_firing01(g)
    nc = np.clip(n, 0.0, 1.0)
    rgba_s = grid_to_rgba(g, colormap="spectral")
    np.testing.assert_allclose(rgba_s[..., 0], nc, rtol=0, atol=1e-6)


@pytest.mark.parametrize("cmap", ["spectral", "diverging", "biphasic"])
def test_grid_to_rgba_uniform_nonzero_not_invisible(cmap: str):
    """Uniform non-zero activity stays visible (same n pipeline as firing: g/max -> ones)."""
    g = np.full((32, 32), 0.37, dtype=np.float32)
    kwargs = {}
    if cmap == "biphasic":
        kwargs["biphasic_center"] = 0.0
    rgba = grid_to_rgba(g, colormap=cmap, **kwargs)  # type: ignore[arg-type]
    assert rgba.shape == (32, 32, 4)
    assert float(np.mean(rgba[..., 3])) > 0.15
    assert float(np.max(rgba[..., 3])) > 0.15


def test_grid_to_rgba_spectral_uniform_zero_stays_dark():
    """Truly silent layer (all zeros) should stay dark."""
    g = np.zeros((16, 16), dtype=np.float32)
    rgba = grid_to_rgba(g, colormap="spectral")
    assert float(np.max(rgba[..., 3])) < 1e-5


def test_grid_to_rgba_pre_normalized_uniform_like_cone_tile():
    """Values in [0,1] spatially flat: n = g/max = 1 everywhere (same as amber)."""
    g = np.full((64, 64), 0.25, dtype=np.float32)
    for cmap in ("spectral", "diverging"):
        rgba = grid_to_rgba(g, colormap=cmap)  # type: ignore[arg-type]
        assert float(np.max(rgba[..., 3])) > 0.1, cmap


def test_grid_to_rgba_diverging():
    """Diverging colormap runs without error."""
    g = np.array([[-1.0, 0.0], [1.0, 0.5]], dtype=np.float32)
    rgba = grid_to_rgba(g, colormap="diverging")
    assert rgba.shape == (2, 2, 4)


def test_grid_to_rgba_unknown_raises():
    """Unknown colormap raises ValueError."""
    with pytest.raises(ValueError, match="Unknown colormap"):
        grid_to_rgba(np.zeros((2, 2)), colormap="invalid")  # type: ignore[arg-type]
