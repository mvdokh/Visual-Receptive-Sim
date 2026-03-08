"""Tests for simulation/layers: cones, horizontal, bipolar, amacrine, rgc helpers."""
from __future__ import annotations

import numpy as np
import pytest

from src.config import default_config
from src.simulation.layers.cones import compute_cone_responses
from src.simulation.stimulus.spectral import build_stimulus_spectrum


def test_compute_cone_responses_shape():
    """Cone responses are (H, W) each."""
    cfg = default_config()
    spec = build_stimulus_spectrum(
        {"type": "full_field", "intensity": 1.0},
        cfg.spectral,
        (24, 32),
    )
    L, M, S = compute_cone_responses(spec, cfg.spectral)
    assert L.shape == (24, 32)
    assert M.shape == (24, 32)
    assert S.shape == (24, 32)


def test_compute_cone_responses_positive_for_light():
    """Positive spectrum gives positive L, M, S."""
    cfg = default_config()
    spec = np.ones((8, 8, cfg.spectral.wavelengths.size), dtype=np.float32) * 0.5
    L, M, S = compute_cone_responses(spec, cfg.spectral)
    assert np.all(L >= 0) and np.all(M >= 0) and np.all(S >= 0)
    assert float(np.mean(L)) > 0
