"""Tests for rendering utilities that do not require GPU/display."""
from __future__ import annotations

import numpy as np
import pytest

from src.config import signal_flow_slab_layout
from src.rendering.scene_3d.layer_trace_strips import allocate_trace_buffers, TRACE_TIME_STEPS
from src.rendering.scene_3d.connectivity_lines import _y_center


def test_allocate_trace_buffers_shape():
    """allocate_trace_buffers returns (num_layers, slab_height_px, T)."""
    buf = allocate_trace_buffers(6, slab_height_px=64)
    assert buf.shape == (6, 64, TRACE_TIME_STEPS)
    assert buf.dtype == np.float32
    assert np.all(buf == 0)


def test_y_center_from_layout():
    """_y_center returns y_top - thickness/2 for known layer."""
    layout = list(signal_flow_slab_layout())
    y = _y_center(layout, "RGC")
    assert y == pytest.approx(2.6 - 0.5 / 2.0)  # 2.35 from layout (2.6, 0.5)


def test_y_center_unknown_returns_default():
    """_y_center returns 3.0 for unknown layer name."""
    layout = list(signal_flow_slab_layout())
    assert _y_center(layout, "UnknownLayer") == 3.0
