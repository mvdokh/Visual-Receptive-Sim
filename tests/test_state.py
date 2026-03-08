"""Tests for SimState: grid_shape, ensure_initialized, dirty_flags, smoothed."""
from __future__ import annotations

import numpy as np
import pytest

from src.config import default_config
from src.simulation.state import SimState


def test_grid_shape_matches_config():
    """grid_shape() returns (grid_resolution, grid_resolution)."""
    cfg = default_config()
    cfg.retina.grid_resolution = 128
    state = SimState(config=cfg)
    assert state.grid_shape() == (128, 128)


def test_ensure_initialized_smoothed_and_dirty():
    """ensure_initialized populates smoothed and dirty_flags for all SMOOTHED layers."""
    state = SimState(config=default_config())
    state.ensure_initialized()
    from src.simulation.pipeline import SMOOTHED_LAYERS
    for name in SMOOTHED_LAYERS:
        assert name in state.smoothed
        assert state.smoothed[name].shape == state.grid_shape()
        assert name in state.dirty_flags


def test_ensure_initialized_idempotent():
    """Calling ensure_initialized twice does not change shape or break state."""
    state = SimState(config=default_config())
    state.ensure_initialized()
    h, w = state.grid_shape()
    state.ensure_initialized()
    assert state.cone_L.shape == (h, w)
    assert state.fr_midget_on_L.shape == (h, w)
