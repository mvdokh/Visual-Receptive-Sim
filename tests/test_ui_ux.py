"""Tests for config, state, export, and UI-related invariants (no display required)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.config import default_config, signal_flow_slab_layout
from src.simulation.state import SimState
from src.simulation.stimulus.spectral import build_stimulus_spectrum
from src.gui.panels.data_export import (
    export_screenshot_png,
    export_layer_grids_csv,
    export_layer_grids_npy,
)


def test_default_config_builds():
    """Default config and all sub-configs construct without error."""
    cfg = default_config()
    assert cfg.retina.grid_resolution > 0
    assert cfg.spectral.wavelengths.size > 0
    assert cfg.spectral.sens_L.size == cfg.spectral.wavelengths.size


def test_state_ensure_initialized(state):
    """SimState.ensure_initialized() allocates all layer grids with correct shape."""
    state.ensure_initialized()
    h, w = state.grid_shape()
    assert state.cone_L is not None and state.cone_L.shape == (h, w)
    assert state.fr_midget_on_L is not None and state.fr_midget_on_L.shape == (h, w)


def test_slab_layout_order():
    """Signal flow slab order is Stimulus -> Cones -> Horizontal -> Bipolar -> Amacrine -> RGC."""
    layout = signal_flow_slab_layout()
    names = [t[0] for t in layout]
    assert names == ["Stimulus", "Cones", "Horizontal", "Bipolar", "Amacrine", "RGC"]
    for _, y_top, thick in layout:
        assert thick > 0
        assert y_top > 0


def test_stimulus_types_return_correct_shape(cfg):
    """All documented stimulus types return (H, W, n_wavelengths)."""
    h, w = 32, 32
    nwl = cfg.spectral.wavelengths.size
    types = ["spot", "full_field", "annulus", "bar", "grating", "checkerboard", "image"]
    for stim_type in types:
        params = {"type": stim_type, "intensity": 0.5}
        if stim_type == "image":
            params["image_mask"] = np.zeros((h, w, 3), dtype=np.float32)  # black image
        spec = build_stimulus_spectrum(params, cfg.spectral, (h, w))
        assert spec.shape == (h, w, nwl), f"type={stim_type}"


def test_export_screenshot_png(state):
    """Export PNG does not crash; produces a file."""
    state.ensure_initialized()
    rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = Path(f.name)
    try:
        export_screenshot_png(rgba, path)
        assert path.exists() and path.stat().st_size > 0
    finally:
        path.unlink(missing_ok=True)


def test_export_layer_grids_csv(state):
    """Export CSV does not crash; file contains layer names."""
    state.ensure_initialized()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = Path(f.name)
    try:
        export_layer_grids_csv(state, path)
        text = path.read_text()
        assert "layer" in text and "cone_L" in text
    finally:
        path.unlink(missing_ok=True)


def test_export_layer_grids_npy(state):
    """Export NPY directory creates .npy files for layer grids."""
    state.ensure_initialized()
    with tempfile.TemporaryDirectory() as d:
        dirpath = Path(d)
        export_layer_grids_npy(state, dirpath)
        npy_files = list(dirpath.glob("*.npy"))
        assert len(npy_files) >= 5
        for f in npy_files:
            arr = np.load(f)
            assert arr.ndim == 2
