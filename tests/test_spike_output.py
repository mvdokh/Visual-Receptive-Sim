"""Spike output layer from firing rates."""
from __future__ import annotations

import numpy as np
import pytest

from src.config import default_config
from src.simulation.pipeline import tick
from src.simulation.state import SimState


@pytest.fixture
def state():
    return SimState(config=default_config())


def test_spikes_disabled_zeros(state):
    state.ensure_initialized()
    state.config.spike_output.enabled = False
    state.stimulus_params = {"type": "full_field", "intensity": 0.5}
    tick(state, 0.05)
    assert state.spike_midget_on_L is not None
    assert float(np.max(state.spike_midget_on_L)) == 0.0


def test_spikes_enabled_deterministic_seed(state):
    state.ensure_initialized()
    spo = state.config.spike_output
    spo.enabled = True
    spo.seed = 12345
    spo.use_smoothed_rates = True
    state.spike_rng = None
    state.stimulus_params = {"type": "full_field", "intensity": 0.9}
    for _ in range(50):
        tick(state, 0.05)
    a = state.spike_midget_on_L.copy()

    st2 = SimState(config=default_config())
    st2.ensure_initialized()
    st2.config.spike_output.enabled = True
    st2.config.spike_output.seed = 12345
    st2.config.spike_output.use_smoothed_rates = True
    st2.stimulus_params = {"type": "full_field", "intensity": 0.9}
    for _ in range(50):
        tick(st2, 0.05)
    b = st2.spike_midget_on_L
    assert np.array_equal(a, b)


def test_spike_mean_rate_ordering(state):
    """Higher stimulus intensity → higher time-averaged spike probability (weak check)."""
    state.ensure_initialized()
    state.config.spike_output.enabled = True
    state.config.spike_output.seed = 999
    state.config.spike_output.use_smoothed_rates = True
    state.spike_rng = None
    state.stimulus_params = {"type": "full_field", "intensity": 0.3}
    for _ in range(80):
        tick(state, 0.05)
    lo = float(np.mean(state.spike_midget_on_L))

    state2 = SimState(config=default_config())
    state2.ensure_initialized()
    state2.config.spike_output.enabled = True
    state2.config.spike_output.seed = 999
    state2.config.spike_output.use_smoothed_rates = True
    state2.stimulus_params = {"type": "full_field", "intensity": 0.95}
    for _ in range(80):
        tick(state2, 0.05)
    hi = float(np.mean(state2.spike_midget_on_L))
    assert hi >= lo
