"""
3D viewer benchmarks (CPU + memory smoke tests).

Run with:
  python tests/bench_3d_viewer.py
  pytest tests/bench_3d_viewer.py -v -s

Notes:
- Skips Vispy render benchmarks if vispy isn't installed.
- On large grids (2048²), the goal is to confirm we do not allocate multi-GB
  geometry just to open the viewer.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

# Allow running as script from repo root: python tests/bench_3d_viewer.py
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import default_config, large_field_config
from src.simulation import SimState
from src.viewers.viewer_3d import HAS_VISPY, VispyViewer3D


def _timed(repeat: int, fn: Callable[[], None]) -> tuple[float, float]:
    """Run fn() repeat times; return (mean_ms, std_ms)."""
    times_ms = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times_ms)), float(np.std(times_ms))


def _rss_mb() -> float:
    """
    Best-effort RSS in MB (macOS-friendly). If not available, returns -1.
    """
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On macOS ru_maxrss is bytes; on Linux it's KB.
        # Heuristic: treat very large numbers as bytes.
        if rss > 10_000_000:
            return float(rss) / (1024.0 * 1024.0)
        return float(rss) / 1024.0
    except Exception:
        return -1.0


def _state(cfg) -> SimState:
    s = SimState(config=cfg)
    s.stimulus_params.update({"type": "spot", "intensity": 1.0, "x_deg": 0.0, "y_deg": 0.0})
    s.ensure_initialized()
    return s


def test_bench_3d_update_frame_256():
    cfg = default_config()
    cfg.cell_subsample = 8
    s = _state(cfg)

    if not HAS_VISPY:
        if pytest is not None:
            pytest.skip("vispy not installed")
        return

    viewer = VispyViewer3D(size=(512, 512), config=cfg)

    mean_ms, std_ms = _timed(10, lambda: viewer.update_frame(s))
    print(f"  3D update_frame(256, subsample=8): {mean_ms:7.2f} ± {std_ms:5.2f} ms   RSS~{_rss_mb():.0f} MB")
    if pytest is not None:
        assert mean_ms < 200, "update_frame(256) should be comfortably under a few hundred ms"


def test_bench_3d_render_256():
    cfg = default_config()
    cfg.cell_subsample = 8
    s = _state(cfg)

    if not HAS_VISPY:
        if pytest is not None:
            pytest.skip("vispy not installed")
        return

    viewer = VispyViewer3D(size=(512, 512), config=cfg)
    viewer.update_frame(s)
    mean_ms, std_ms = _timed(10, lambda: viewer.render())
    print(f"  3D render(512x512):               {mean_ms:7.2f} ± {std_ms:5.2f} ms   RSS~{_rss_mb():.0f} MB")


def test_bench_3d_open_large_field_geometry_only():
    """
    Large-field smoke test: construct state + open viewer + update once.
    This should NOT blow up RAM anymore.
    """
    cfg = large_field_config()
    cfg.cell_subsample = 8
    s = _state(cfg)

    if not HAS_VISPY:
        if pytest is not None:
            pytest.skip("vispy not installed")
        return

    before = _rss_mb()
    viewer = VispyViewer3D(size=(512, 512), config=cfg)
    viewer.update_frame(s)
    after = _rss_mb()
    print(f"  3D open+update(2048, subsample=8): RSS {before:.0f} -> {after:.0f} MB")


if __name__ == "__main__":  # pragma: no cover
    print("=== 3D viewer benchmarks ===")
    try:
        test_bench_3d_update_frame_256()
        test_bench_3d_render_256()
        test_bench_3d_open_large_field_geometry_only()
    except Exception as e:
        print(f"Error: {e}")
