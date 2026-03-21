"""Tests for config module: geometry, spectral, layout, connectivity."""
from __future__ import annotations

import pytest

from src.config import (
    default_config,
    large_field_config,
    layer_z_positions,
    signal_flow_slab_layout,
    RetinaGeometry,
    ConnectivityWeights,
    GlobalConfig,
)


def test_retina_geometry_dx_deg():
    """dx_deg = field_size_deg / grid_resolution."""
    r = RetinaGeometry(field_size_deg=1.0, grid_resolution=256)
    assert r.dx_deg == pytest.approx(1.0 / 256)


def test_layer_z_positions_length():
    """layer_z_positions returns 7 values (legacy stack)."""
    z = layer_z_positions()
    assert len(z) == 7
    assert all(isinstance(v, (int, float)) for v in z)


def test_signal_flow_slab_layout_tuples():
    """Each slab is (name, y_top, thickness) with positive thickness."""
    layout = signal_flow_slab_layout()
    assert len(layout) == 6
    for name, y_top, thick in layout:
        assert isinstance(name, str)
        assert y_top > 0
        assert thick > 0


def test_connectivity_weights_defaults():
    """All connectivity weights default to 1.0."""
    cw = ConnectivityWeights()
    assert cw.cone_to_horizontal == 1.0
    assert cw.bipolar_to_rgc == 1.0


def test_global_config_has_all_subconfigs():
    """GlobalConfig has retina, temporal, spectral, horizontal, bipolar, amacrine, dendritic, rgc_nl."""
    cfg = default_config()
    assert hasattr(cfg, "retina") and cfg.retina is not None
    assert hasattr(cfg, "spectral") and cfg.spectral.wavelengths is not None
    assert hasattr(cfg, "horizontal") and cfg.horizontal.alpha_lm == 0.7
    assert hasattr(cfg, "connectivity_weights")
    assert hasattr(cfg, "rgc_population") and cfg.rgc_population.enabled is False


def test_retina_geometry_microns_and_large_field():
    """RetinaGeometry has microns_per_px and grid_size_degrees_physical."""
    r = RetinaGeometry(field_size_deg=1.0, grid_resolution=256, microns_per_px=4.0)
    assert r.microns_per_px == 4.0
    assert r.grid_size_microns == 256 * 4.0
    cfg = large_field_config()
    assert cfg.retina.grid_resolution >= 1024
    assert cfg.retina.field_size_deg > 10.0
