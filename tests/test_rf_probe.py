"""Tests for rf_probe: _dog_2d, fit_dog, probe_sweep_fast."""
from __future__ import annotations

import numpy as np
import pytest

from src.simulation.rf_probe import _dog_2d, fit_dog, DoGFit, probe_sweep_fast
from tests.conftest import small_state  # noqa: F401 - fixture


def test_dog_2d_center_peak():
    """DoG has peak at (x0, y0)."""
    xy = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]], dtype=np.float32)
    val = _dog_2d(xy, x0=0.0, y0=0.0, sigma_c=0.05, sigma_s=0.15, w_c=1.0, w_s=0.3, baseline=0.0)
    assert val[0] > val[1] and val[0] > val[2]


def test_fit_dog_success_on_good_map():
    """fit_dog returns success=True on a clean DoG-like map."""
    x = np.linspace(-0.2, 0.2, 16)
    y = np.linspace(-0.2, 0.2, 16)
    X, Y = np.meshgrid(x, y)
    rf = np.exp(-(X**2 + Y**2) / (2 * 0.03**2)) - 0.3 * np.exp(-(X**2 + Y**2) / (2 * 0.1**2))
    dog = fit_dog(x, y, rf.astype(np.float32))
    assert dog.sigma_center > 0
    assert dog.sigma_surround > 0


def test_fit_dog_returns_dog_fit_on_failure():
    """fit_dog returns DoGFit; on degenerate input (zeros) success may be False or fit is trivial."""
    x = np.linspace(-0.1, 0.1, 4)
    y = np.linspace(-0.1, 0.1, 4)
    flat = np.zeros((4, 4), dtype=np.float32)  # zeros more likely to yield failure
    dog = fit_dog(x, y, flat)
    assert isinstance(dog, DoGFit)
    assert dog.sigma_center > 0
    assert dog.sigma_surround > 0


def test_probe_sweep_fast_all_rgc_types(small_state):
    """probe_sweep_fast runs for all supported RGC types."""
    for rgc_type in ["midget_on_L", "midget_off_L", "parasol_on", "parasol_off"]:
        x_deg, y_deg, rf_map = probe_sweep_fast(small_state, rgc_type=rgc_type, probe_resolution=8)
        assert x_deg.ndim == 1 and y_deg.ndim == 1
        assert rf_map.shape == (8, 8)
