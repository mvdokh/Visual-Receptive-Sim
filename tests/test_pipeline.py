"""Tests for pipeline.tick: multi-step, connectivity weights, all state attributes."""
from __future__ import annotations

import numpy as np
import pytest

from src.config import default_config
from src.simulation.pipeline import tick, SMOOTHED_LAYERS
from src.simulation.state import SimState


@pytest.fixture
def state():
    return SimState(config=default_config())


def test_tick_populates_all_layer_arrays(state):
    """After tick, cone_L/M/S, h_activation, bipolar, amacrine, fr_*, lm_opponent, by_opponent are set."""
    state.ensure_initialized()
    state.stimulus_params = {"type": "full_field", "intensity": 0.5}
    tick(state, 0.05)
    assert state.cone_L is not None and state.cone_L.shape == state.grid_shape()
    assert state.h_activation is not None
    assert state.bp_midget_on_L is not None
    assert state.amacrine_aii is not None
    assert state.fr_midget_on_L is not None
    assert state.lm_opponent is not None
    assert state.by_opponent is not None


def test_tick_increments_time(state):
    """state.time increases by dt."""
    state.ensure_initialized()
    t0 = state.time
    tick(state, 0.03)
    assert state.time == pytest.approx(t0 + 0.03)
    tick(state, 0.02)
    assert state.time == pytest.approx(t0 + 0.05)


def test_tick_connectivity_weights_scale_response(state):
    """With bipolar_to_rgc=0 from cold start, RGC drive is 0 so LN gives half-max (r_max/2)."""
    state.ensure_initialized()
    state.config.connectivity_weights.bipolar_to_rgc = 0.0
    state.stimulus_params = {"type": "full_field", "intensity": 0.0}  # no stimulus
    for _ in range(30):
        tick(state, 0.05)
    # Zero drive → sigmoid(0) = r_max/2 = 60 with default x_half=0
    assert float(np.mean(state.fr_midget_on_L)) == pytest.approx(60.0, abs=2.0)


def test_cone_to_horizontal_scales_horizontal_activation(state):
    """cone_to_horizontal multiplies cone drive into the horizontal pool; 0 suppresses h_activation."""
    state.ensure_initialized()
    state.stimulus_params = {"type": "full_field", "intensity": 0.8}
    state.config.connectivity_weights.cone_to_horizontal = 0.0
    tick(state, 0.05)
    h_zero = float(np.mean(state.h_activation))
    state.config.connectivity_weights.cone_to_horizontal = 1.5
    tick(state, 0.05)
    h_scaled = float(np.mean(state.h_activation))
    assert h_zero < 1e-3
    assert h_scaled > h_zero


def test_rgc_nl_r_max_affects_firing_ceiling(state):
    """Cell NL r_max is applied in the pipeline; lowering it reduces mean RGC firing under drive."""
    state.ensure_initialized()
    state.stimulus_params = {"type": "full_field", "intensity": 0.9}
    state.config.rgc_nl.r_max = 120.0
    for _ in range(40):
        tick(state, 0.05)
    hi = float(np.mean(state.fr_midget_on_L))
    state.config.rgc_nl.r_max = 40.0
    for _ in range(40):
        tick(state, 0.05)
    lo = float(np.mean(state.fr_midget_on_L))
    assert hi > lo + 5.0


def test_rgc_population_enabled_skew_changes_horizontal_activation(state):
    """Uniform group scales normalize away; skewing one group changes modulation and h_activation."""
    state.ensure_initialized()
    state.stimulus_params = {"type": "full_field", "intensity": 0.6}
    rpc = state.config.rgc_population
    rpc.enabled = False
    tick(state, 0.05)
    h_disabled = float(np.mean(state.h_activation))
    rpc.enabled = True
    for g in rpc.group_scales:
        rpc.group_scales[g] = 1.0
    rpc.group_scales["OFF_sustained"] = 5.0
    tick(state, 0.05)
    h_skew = float(np.mean(state.h_activation))
    assert h_disabled != h_skew


def test_tick_smoothed_layers_updated(state):
    """After tick, state attributes for SMOOTHED_LAYERS match state.smoothed (pipeline overwrites with smoothed)."""
    state.ensure_initialized()
    state.stimulus_params = {"type": "spot", "intensity": 1.0}
    tick(state, 0.05)
    for attr in SMOOTHED_LAYERS:
        arr = getattr(state, attr, None)
        if arr is not None and attr in state.smoothed:
            np.testing.assert_array_almost_equal(arr, state.smoothed[attr])
