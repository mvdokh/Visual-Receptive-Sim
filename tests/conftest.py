"""Pytest fixtures for RGC Circuit Simulator tests."""
from __future__ import annotations

import numpy as np
import pytest

from src.config import default_config
from src.simulation.state import SimState


@pytest.fixture
def cfg():
    """Default global config (cone fundamentals may be CSV or Gaussian fallback)."""
    return default_config()


@pytest.fixture
def state(cfg):
    """SimState with default config; not yet ticked."""
    return SimState(config=cfg)


@pytest.fixture
def small_state(cfg):
    """SimState on a small grid for faster tests."""
    cfg.retina.grid_resolution = 64
    cfg.retina.field_size_deg = 0.5
    return SimState(config=cfg)
