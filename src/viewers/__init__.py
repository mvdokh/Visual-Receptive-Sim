"""
3D viewer module for RGC Circuit Simulator.

Vispy-based 3D viewer; renders to buffer for embedding in the main app (Dear PyGui).
PyMOL-style camera, circuit tracing, scale bar.
"""

from src.viewers.viewer_3d import HAS_VISPY, VispyViewer3D
from src.viewers.layer_manager import LayerManager
from src.viewers.circuit_tracer import CircuitTracer, CircuitTree

__all__ = [
    "HAS_VISPY",
    "VispyViewer3D",
    "LayerManager",
    "CircuitTracer",
    "CircuitTree",
]
