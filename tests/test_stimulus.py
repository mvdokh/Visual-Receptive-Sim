"""Tests for stimulus/spectral: all stimulus types, edge cases, params."""
from __future__ import annotations

import numpy as np
import pytest

from src.simulation.stimulus.spectral import build_stimulus_spectrum
from src.config import default_config


@pytest.fixture
def cfg():
    return default_config()


def test_stimulus_default_params_none(cfg):
    """params=None uses defaults (spot-like)."""
    spec = build_stimulus_spectrum(None, cfg.spectral, (32, 32))
    assert spec.shape == (32, 32, cfg.spectral.wavelengths.size)
    assert np.any(spec > 0)


def test_stimulus_spectral_none_uses_default_config():
    """spectral=None uses default_config().spectral."""
    spec = build_stimulus_spectrum({"type": "full_field"}, None, (16, 16))
    assert spec.shape == (16, 16, 65)  # 380:5:700 -> 65 points


def test_stimulus_annulus_ring(cfg):
    """Annulus has power in ring, less at center."""
    params = {"type": "annulus", "radius_deg": 0.2, "inner_radius_deg": 0.08, "intensity": 1.0}
    spec = build_stimulus_spectrum(params, cfg.spectral, (64, 64))
    total = np.sum(spec, axis=-1)
    center = total[32, 32]
    # Ring around mid-radius
    r_mid = 24  # pixel offset ~0.15 deg
    ring_val = total[32, 32 + r_mid]
    assert ring_val > center


def test_stimulus_bar(cfg):
    """Bar stimulus returns correct shape and accepts width/orientation params."""
    params = {"type": "bar", "width_deg": 0.08, "orientation_deg": 0.0, "intensity": 1.0}
    spec = build_stimulus_spectrum(params, cfg.spectral, (64, 64))
    assert spec.shape == (64, 64, cfg.spectral.wavelengths.size)
    assert np.any(spec > 0)
    params["orientation_deg"] = 90.0
    spec90 = build_stimulus_spectrum(params, cfg.spectral, (64, 64))
    assert spec90.shape == spec.shape


def test_stimulus_grating_alternating(cfg):
    """Grating has alternating high/low power along direction."""
    params = {"type": "grating", "spatial_freq_cpd": 2.0, "phase_deg": 0.0, "intensity": 1.0}
    spec = build_stimulus_spectrum(params, cfg.spectral, (32, 32))
    total = np.sum(spec, axis=-1)
    assert np.min(total) < np.max(total)


def test_stimulus_moving_spot_time(cfg):
    """Moving spot position changes with time_s."""
    params = {"type": "moving_spot", "radius_deg": 0.1, "vx_deg_s": 0.5, "intensity": 1.0}
    spec0 = build_stimulus_spectrum(params, cfg.spectral, (64, 64), time_s=0.0)
    spec1 = build_stimulus_spectrum(params, cfg.spectral, (64, 64), time_s=0.2)
    assert not np.allclose(spec0, spec1)


def test_stimulus_dual_spot_two_colors(cfg):
    """Dual spot returns spectrum with two spectral contributions."""
    params = {
        "type": "dual_spot",
        "wavelength_nm": 550.0,
        "wavelength2_nm": 450.0,
        "intensity": 1.0,
        "intensity2": 0.8,
        "radius_deg": 0.08,
        "x_deg": -0.1,
        "y_deg": 0.0,
        "x2_deg": 0.1,
        "y2_deg": 0.0,
    }
    spec = build_stimulus_spectrum(params, cfg.spectral, (64, 64))
    assert spec.shape == (64, 64, cfg.spectral.wavelengths.size)
    assert np.any(spec > 0)


def test_stimulus_image_none_mask_returns_zeros(cfg):
    """Image type with no image_mask returns zero spectrum (early return)."""
    params = {"type": "image", "intensity": 1.0}
    spec = build_stimulus_spectrum(params, cfg.spectral, (16, 16))
    assert np.all(spec == 0)


def test_stimulus_unknown_type_falls_back_to_spot(cfg):
    """Unknown type falls back to spot-like mask."""
    params = {"type": "unknown_type", "radius_deg": 0.1, "intensity": 1.0}
    spec = build_stimulus_spectrum(params, cfg.spectral, (32, 32))
    assert spec.shape == (32, 32, cfg.spectral.wavelengths.size)
    assert np.any(spec > 0)
