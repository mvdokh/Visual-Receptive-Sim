"""Tests for biological assumptions (cone fundamentals, center-surround, ON/OFF, LN, temporal, stimulus, DoG)."""
from __future__ import annotations

import numpy as np
import pytest

from src.config import default_config
from src.simulation.fast_conv import gaussian_pool_2d
from src.simulation.fast_layers import sigmoid_ln, temporal_rc
from src.simulation.pipeline import tick
from src.simulation.rf_probe import fit_dog, probe_sweep_fast
from src.simulation.state import SimState
from src.simulation.stimulus.spectral import build_stimulus_spectrum


# ---- Cone fundamentals (Stockman & Sharpe style) ----
def test_cone_fundamentals_normalized(cfg):
    """L/M/S sensitivities are normalized to max 1."""
    sp = cfg.spectral
    assert float(np.max(sp.sens_L)) == pytest.approx(1.0)
    assert float(np.max(sp.sens_M)) == pytest.approx(1.0)
    assert float(np.max(sp.sens_S)) == pytest.approx(1.0)


def test_cone_fundamentals_peak_wavelengths(cfg):
    """L/M/S peaks lie in expected wavelength bands (L long, M mid, S short)."""
    wl = cfg.spectral.wavelengths
    L_peak = float(wl[np.argmax(cfg.spectral.sens_L)])
    M_peak = float(wl[np.argmax(cfg.spectral.sens_M)])
    S_peak = float(wl[np.argmax(cfg.spectral.sens_S)])
    assert L_peak > M_peak > S_peak
    assert 500 <= L_peak <= 620
    assert 480 <= M_peak <= 580
    assert 380 <= S_peak <= 480


def test_cone_integration_narrowband(state):
    """Narrowband stimulus at 550 nm gives L, M > S (L/M overlap at mid wavelength)."""
    state.ensure_initialized()
    state.stimulus_params = {"type": "full_field", "wavelength_nm": 550.0, "intensity": 1.0}
    tick(state, 0.05)
    mean_L = float(np.mean(state.cone_L))
    mean_M = float(np.mean(state.cone_M))
    mean_S = float(np.mean(state.cone_S))
    assert mean_L > 0 and mean_M > 0
    assert mean_L > mean_S and mean_M > mean_S


# ---- Center-surround (horizontal feedback) ----
def test_center_surround_subtraction(state):
    """Effective cone output is reduced by horizontal feedback (surround subtraction)."""
    state.ensure_initialized()
    state.stimulus_params = {"type": "full_field", "wavelength_nm": 550.0, "intensity": 1.0}
    tick(state, 0.05)
    # With uniform stimulus, H is positive; C_eff = C - alpha*H so C_eff < C
    assert np.all(state.cone_L_eff <= state.cone_L + 1e-6)
    assert np.all(state.cone_M_eff <= state.cone_M + 1e-6)


# ---- ON/OFF bipolar rectification ----
def test_bipolar_on_off_rectification(state):
    """ON path is max(0, drive); OFF path is max(0, -drive). Positive drive -> ON>0, OFF~0."""
    state.ensure_initialized()
    state.stimulus_params = {"type": "spot", "wavelength_nm": 550.0, "intensity": 1.0, "radius_deg": 0.2}
    tick(state, 0.05)
    # Center of spot: cone_eff positive -> bp_midget_on_L > 0, bp_midget_off_L ~ 0
    cy, cx = state.grid_shape()[0] // 2, state.grid_shape()[1] // 2
    assert state.bp_midget_on_L[cy, cx] >= 0
    assert state.bp_midget_off_L[cy, cx] >= 0
    assert state.bp_midget_on_L[cy, cx] > state.bp_midget_off_L[cy, cx]


# ---- LN sigmoid (firing rate) ----
def test_ln_sigmoid_half_max():
    """At x = x_half, output is r_max/2."""
    x = np.array([0.0], dtype=np.float32)
    r = sigmoid_ln(x, r_max=100.0, x_half=0.0, slope=4.0)
    assert float(r[0]) == pytest.approx(50.0, rel=0.01)


def test_ln_sigmoid_asymptotes():
    """Large positive input -> r_max; large negative -> 0."""
    r_max = 120.0
    high = sigmoid_ln(np.array([100.0], dtype=np.float32), r_max, 0.0, 4.0)
    low = sigmoid_ln(np.array([-100.0], dtype=np.float32), r_max, 0.0, 4.0)
    assert float(high[0]) == pytest.approx(r_max, rel=0.01)
    assert float(low[0]) == pytest.approx(0.0, abs=0.01)


# ---- Temporal RC ----
def test_temporal_rc_alpha_one():
    """Alpha=1: state becomes target in one step."""
    state_arr = np.array([0.0, 1.0], dtype=np.float32)
    target = np.array([2.0, 3.0], dtype=np.float32)
    temporal_rc(state_arr, target, 1.0)
    np.testing.assert_array_almost_equal(state_arr, target)


def test_temporal_rc_alpha_zero():
    """Alpha=0: state unchanged."""
    state_arr = np.array([1.0, 2.0], dtype=np.float32)
    target = np.array([10.0, 20.0], dtype=np.float32)
    temporal_rc(state_arr, target, 0.0)
    np.testing.assert_array_almost_equal(state_arr, [1.0, 2.0])


# ---- Stimulus spectrum ----
def test_stimulus_spot_shape(cfg):
    """Spot stimulus has power inside radius, low outside."""
    params = {"type": "spot", "radius_deg": 0.1, "x_deg": 0.0, "y_deg": 0.0, "intensity": 1.0}
    spec = build_stimulus_spectrum(params, cfg.spectral, (64, 64))
    total = np.sum(spec, axis=-1)
    center = total[32, 32]
    corner = total[0, 0]
    assert center > 0
    assert center > corner


def test_stimulus_full_field_uniform(cfg):
    """Full-field stimulus is spatially uniform."""
    params = {"type": "full_field", "intensity": 1.0}
    spec = build_stimulus_spectrum(params, cfg.spectral, (32, 32))
    total = np.sum(spec, axis=-1)
    assert np.std(total) < 1e-5
    assert np.mean(total) > 0


def test_stimulus_image_rgb_three_band(cfg):
    """Image stimulus with RGB produces spectrum with three wavelength bands (L/M/S input)."""
    h, w = 16, 16
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[8, 8, 0] = 1.0  # red pixel at center
    rgb[8, 0, 1] = 1.0  # green at left
    rgb[0, 8, 2] = 1.0  # blue at top
    params = {"type": "image", "image_mask": rgb, "intensity": 1.0}
    spec = build_stimulus_spectrum(params, cfg.spectral, (h, w))
    assert spec.shape == (h, w, cfg.spectral.wavelengths.size)
    assert np.all(spec >= 0)
    # Red pixel should have more power at long wavelengths
    lam = cfg.spectral.wavelengths
    spec_center = spec[8, 8, :]
    long_power = np.sum(spec_center[lam > 580])
    short_power = np.sum(spec_center[lam < 500])
    assert long_power > short_power


# ---- DoG fit (receptive field) ----
def test_dog_fit_synthetic(small_state):
    """DoG fit on probe-generated RF map returns positive sigma_center and sigma_surround."""
    x_deg, y_deg, rf_map = probe_sweep_fast(small_state, rgc_type="midget_on_L", probe_resolution=12)
    dog = fit_dog(x_deg, y_deg, rf_map)
    assert dog.sigma_center > 0
    assert dog.sigma_surround > 0


# ---- Color opponent ----
def test_lm_opponent_sign(small_state):
    """L-dominated stimulus -> L-M > 0; M-dominated -> L-M < 0 (at least in mean)."""
    small_state.ensure_initialized()
    small_state.stimulus_params = {"type": "full_field", "wavelength_nm": 600.0, "intensity": 1.0}
    tick(small_state, 0.05)
    lm = float(np.mean(small_state.lm_opponent))
    # 600 nm favors L over M
    assert lm > 0
