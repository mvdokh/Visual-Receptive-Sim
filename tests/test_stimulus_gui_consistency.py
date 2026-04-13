"""
Stimulus params ↔ viewport/sim consistency (no Dear PyGui window).

Guards against stale ``stimulus_spectrum`` in the middle panel and broken type switches.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.gui import app as app_module
from src.rendering.heatmap import spectrum_to_stimulus_rgba
from src.simulation.pipeline import tick
from src.simulation.state import SimState
from src.simulation.stimulus.spectral import build_stimulus_spectrum


STIMULUS_COMBO_TYPES = [
    "spot",
    "full_field",
    "annulus",
    "bar",
    "grating",
    "checkerboard",
    "moving_spot",
    "moving_bar",
    "moving_grating",
    "expanding_ring",
    "drifting_grating_full",
    "dual_spot",
    "image",
]


def _minimal_params(stim_type: str, h: int, w: int) -> dict:
    p: dict = {"type": stim_type, "intensity": 0.8, "wavelength_nm": 550.0}
    if stim_type == "image":
        p["image_mask"] = np.full((h, w, 3), 0.3, dtype=np.float32)
    if stim_type in ("spot", "annulus", "moving_spot", "expanding_ring", "dual_spot"):
        p.setdefault("radius_deg", 0.12)
    if stim_type == "annulus":
        p.setdefault("inner_radius_deg", 0.04)
    if stim_type in ("bar", "moving_bar"):
        p.setdefault("width_deg", 0.08)
        p.setdefault("orientation_deg", 15.0)
    if stim_type in ("grating", "moving_grating", "drifting_grating_full"):
        p.setdefault("spatial_freq_cpd", 2.0)
        p.setdefault("phase_deg", 0.0)
        p.setdefault("orientation_deg", 0.0)
    if stim_type == "checkerboard":
        p.setdefault("width_deg", 0.1)
    if stim_type in ("moving_spot", "moving_bar", "moving_grating"):
        p.setdefault("vx_deg_s", 0.2)
        p.setdefault("vy_deg_s", 0.0)
        p.setdefault("motion_mode", "linear")
    if stim_type == "dual_spot":
        p.setdefault("x_deg", -0.05)
        p.setdefault("y_deg", 0.0)
        p.setdefault("x2_deg", 0.05)
        p.setdefault("y2_deg", 0.0)
        p.setdefault("radius2_deg", 0.1)
        p.setdefault("wavelength2_nm", 450.0)
        p.setdefault("intensity2", 0.7)
    return p


def test_render_stimulus_rgba_uses_params_not_cached_spectrum(state: SimState) -> None:
    """Viewport stimulus must follow ``stimulus_params``, not a stale ``stimulus_spectrum``."""
    state.ensure_initialized()
    h, w = state.grid_shape()
    cfg = state.config
    # Pretend last tick left a bright grating in state.stimulus_spectrum
    fake = build_stimulus_spectrum(
        {"type": "grating", "intensity": 1.0, "spatial_freq_cpd": 2.0, "phase_deg": 0.0},
        cfg.spectral,
        (h, w),
        retina=cfg.retina,
    )
    state.stimulus_spectrum = fake
    # User switched UI to a small off-center spot (params) without running tick
    state.stimulus_params = {
        "type": "spot",
        "intensity": 1.0,
        "radius_deg": 0.05,
        "x_deg": 0.2,
        "y_deg": -0.15,
        "wavelength_nm": 550.0,
    }
    rgba = app_module._render_stimulus_rgba(state)
    ref_spec = build_stimulus_spectrum(
        state.stimulus_params,
        cfg.spectral,
        (h, w),
        time_s=float(state.time),
        retina=cfg.retina,
    )
    ref_rgba = spectrum_to_stimulus_rgba(ref_spec, cfg.spectral.wavelengths)
    assert rgba.shape == ref_rgba.shape
    assert np.allclose(rgba, ref_rgba, rtol=1e-5, atol=1e-5)
    assert not np.allclose(
        spectrum_to_stimulus_rgba(fake, cfg.spectral.wavelengths),
        ref_rgba,
        atol=0.02,
    )


def test_render_stimulus_image_type_without_mask_uses_fresh_spectrum(state: SimState) -> None:
    """Selecting image without a mask should not show a previous non-image spectrum."""
    state.ensure_initialized()
    h, w = state.grid_shape()
    cfg = state.config
    state.stimulus_spectrum = build_stimulus_spectrum(
        {"type": "full_field", "intensity": 1.0},
        cfg.spectral,
        (h, w),
        retina=cfg.retina,
    )
    state.stimulus_params = {"type": "image", "intensity": 1.0}
    rgba = app_module._render_stimulus_rgba(state)
    assert rgba.shape == (h, w, 4)
    assert float(np.max(rgba[..., :3])) < 1e-5


def test_all_combo_stimulus_types_build_finite_spectrum(cfg) -> None:
    h, w = 48, 48
    nwl = cfg.spectral.wavelengths.size
    for stim_type in STIMULUS_COMBO_TYPES:
        params = _minimal_params(stim_type, h, w)
        spec = build_stimulus_spectrum(
            params,
            cfg.spectral,
            (h, w),
            time_s=0.05,
            retina=cfg.retina,
        )
        assert spec.shape == (h, w, nwl), stim_type
        assert np.all(np.isfinite(spec)), stim_type


def test_sync_stimulus_type_pops_image_mask(state: SimState) -> None:
    state.stimulus_params = {
        "type": "image",
        "image_mask": np.ones((8, 8, 3), dtype=np.float32),
    }
    app_module.sync_stimulus_type_in_params(state, "spot")
    assert state.stimulus_params["type"] == "spot"
    assert "image_mask" not in state.stimulus_params


def test_shared_stimulus_params_double_buffer_sim_states(cfg) -> None:
    """UI and worker share one params dict; edits are visible from both states."""
    a = SimState(config=cfg)
    b = SimState(config=cfg)
    shared: dict = {"type": "spot", "radius_deg": 0.1, "intensity": 1.0}
    a.stimulus_params = shared
    b.stimulus_params = shared
    shared["type"] = "full_field"
    assert a.stimulus_params["type"] == "full_field"
    assert b.stimulus_params["type"] == "full_field"


def test_tick_keeps_stimulus_spectrum_aligned_with_params(state: SimState) -> None:
    """After tick, cached spectrum matches params (pipeline invariant)."""
    state.ensure_initialized()
    h, w = state.grid_shape()
    state.stimulus_params = _minimal_params("bar", h, w)
    tick(state, 1.0 / 60.0)
    assert state.stimulus_spectrum is not None
    fresh = build_stimulus_spectrum(
        state.stimulus_params,
        state.config.spectral,
        (h, w),
        time_s=float(state.time),
        retina=state.config.retina,
    )
    assert np.allclose(state.stimulus_spectrum, fresh, rtol=1e-4, atol=1e-4)


def test_apply_stimulus_type_change_calls_visibility(monkeypatch, state: SimState) -> None:
    seen: list[tuple[str, SimState | None]] = []

    def _rec(t: str, s: SimState | None = None) -> None:
        seen.append((t, s))

    monkeypatch.setattr(app_module, "_update_stimulus_visibility", _rec)
    app_module.apply_stimulus_type_change(state, "grating")
    assert state.stimulus_params["type"] == "grating"
    assert seen == [("grating", state)]


def test_sequential_type_switches_distinct_outputs(cfg) -> None:
    """Rapid type changes produce different spectra (no stuck pattern)."""
    h, w = 32, 32
    prev = None
    # Use a small off-center spot so it cannot match full-field or wide-field patterns.
    specials = {
        "spot": {
            "radius_deg": 0.02,
            "x_deg": 0.15,
            "y_deg": -0.1,
        },
    }
    for stim_type in ("full_field", "spot", "bar", "checkerboard", "annulus"):
        p = _minimal_params(stim_type, h, w)
        p.update(specials.get(stim_type, {}))
        spec = build_stimulus_spectrum(
            p, cfg.spectral, (h, w), time_s=0.0, retina=cfg.retina
        )
        flat = spec.sum(axis=-1)
        if prev is not None:
            assert not np.allclose(flat, prev, atol=1e-3), stim_type
        prev = flat
