"""
Benchmark tests to find performance bottlenecks. Run with:
  python tests/bench_performance.py   # no pytest needed
  pytest tests/bench_performance.py -v -s

Reports mean ± std ms per operation. Target: main-thread frame < 16.67 ms for 60 FPS.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running as script from repo root: python tests/bench_performance.py
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import time
from typing import Callable

import numpy as np

try:
    import pytest
except ImportError:
    pytest = None

from src.config import default_config, large_field_config
from src.simulation import SimState, tick
from src.simulation.cell_positions import CellPositions, pick_nearest_cell_any_layer
from src.simulation.stimulus.spectral import build_stimulus_spectrum
from src.simulation.fast_conv import gaussian_pool_2d
from src.simulation.connectivity import compute_cone_connectivity, compute_rgc_connectivity
from src.rendering.heatmap import block_average_downsample_rgba, grid_to_rgba


def _timed(repeat: int, fn: Callable[[], None], name: str) -> tuple[float, float]:
    """Run fn() repeat times; return (mean_ms, std_ms)."""
    times_ms = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times_ms)), float(np.std(times_ms))


REPEAT = 15
FRAME_BUDGET_MS = 16.67  # 60 FPS


def _state_256():
    cfg = default_config()
    s = SimState(config=cfg)
    s.stimulus_params.update({"type": "spot", "intensity": 1.0, "x_deg": 0.0, "y_deg": 0.0})
    s.ensure_initialized()
    return s


def _state_2048():
    cfg = large_field_config()
    s = SimState(config=cfg)
    s.stimulus_params.update({"type": "spot", "intensity": 1.0, "x_deg": 0.0, "y_deg": 0.0})
    s.ensure_initialized()
    return s


def test_bench_tick_256():
    """Pipeline tick (256x256): must be << 16 ms so worker can run at 60 Hz."""
    state_256 = _state_256()
    mean_ms, std_ms = _timed(REPEAT, lambda: tick(state_256, 1.0 / 60.0), "tick_256")
    print(f"  tick(256):     {mean_ms:7.2f} ± {std_ms:5.2f} ms  (target < {FRAME_BUDGET_MS:.0f} ms for 60 Hz sim)")
    if pytest is not None:
        assert mean_ms < 100, "tick(256) should be well under 100 ms"


def test_bench_tick_2048():
    """Pipeline tick (2048x2048): report only; often > 16 ms."""
    state_2048 = _state_2048()
    mean_ms, std_ms = _timed(REPEAT, lambda: tick(state_2048, 1.0 / 60.0), "tick_2048")
    print(f"  tick(2048):    {mean_ms:7.2f} ± {std_ms:5.2f} ms  (target < {FRAME_BUDGET_MS:.0f} ms for 60 Hz sim)")


def test_bench_render_2d_256():
    """2D render path (256): layer -> grid_to_rgba -> upscale (no downsample)."""
    state_256 = _state_256()
    layer = state_256.fr_midget_on_L if state_256.fr_midget_on_L is not None else np.zeros(state_256.grid_shape(), dtype=np.float32)
    DISPLAY_SCALE = 4
    MAX_DISPLAY_SIDE = 1024

    def frame():
        rgba = grid_to_rgba(layer, colormap="firing")
        if rgba.shape[0] > MAX_DISPLAY_SIDE or rgba.shape[1] > MAX_DISPLAY_SIDE:
            rgba = block_average_downsample_rgba(rgba, MAX_DISPLAY_SIDE)
        else:
            rgba = np.repeat(np.repeat(rgba, DISPLAY_SCALE, axis=0), DISPLAY_SCALE, axis=1)
        tex_data = np.ascontiguousarray(rgba.astype(np.float32)).flatten()
        assert tex_data.size > 0

    mean_ms, std_ms = _timed(REPEAT, frame, "render_2d_256")
    print(f"  render_2d(256): {mean_ms:7.2f} ± {std_ms:5.2f} ms  (target < {FRAME_BUDGET_MS:.0f} ms)")
    if pytest is not None:
        assert mean_ms < FRAME_BUDGET_MS * 2, "2D render 256 should be well under 2 frames"


def test_bench_render_2d_2048():
    """2D render path (2048): layer -> grid_to_rgba -> block_downsample to 1024."""
    state_2048 = _state_2048()
    layer = state_2048.fr_midget_on_L if state_2048.fr_midget_on_L is not None else np.zeros(state_2048.grid_shape(), dtype=np.float32)
    MAX_DISPLAY_SIDE = 1024

    def frame():
        rgba = grid_to_rgba(layer, colormap="firing")
        rgba = block_average_downsample_rgba(rgba, MAX_DISPLAY_SIDE)
        tex_data = np.ascontiguousarray(rgba.astype(np.float32)).flatten()
        assert tex_data.size > 0

    mean_ms, std_ms = _timed(REPEAT, frame, "render_2d_2048")
    print(f"  render_2d(2048): {mean_ms:7.2f} ± {std_ms:5.2f} ms  (target < {FRAME_BUDGET_MS:.0f} ms)")


def test_bench_downsample_rgba_only():
    """Block-average downsample only (2048x2048 RGBA -> 1024)."""
    rgba = np.random.rand(2048, 2048, 4).astype(np.float32)

    def fn():
        block_average_downsample_rgba(rgba, 1024)

    mean_ms, std_ms = _timed(REPEAT, fn, "downsample_rgba_2048")
    print(f"  downsample_rgba(2048->1024): {mean_ms:7.2f} ± {std_ms:5.2f} ms")


def test_bench_stats_numpy_only():
    """Stats-like numpy work (no DPG): mean, std, min, max over layers + sparkline + histogram."""
    state_256 = _state_256()
    layer_data = {
        "Stimulus": np.sum(state_256.stimulus_spectrum, axis=-1) if state_256.stimulus_spectrum is not None else None,
        "Cones L": state_256.cone_L, "Cones M": state_256.cone_M, "Cones S": state_256.cone_S,
        "Horizontal": state_256.h_activation, "Bipolar": state_256.bp_diffuse_on,
        "Amacrine": state_256.amacrine_aii, "RGC": state_256.fr_midget_on_L,
    }

    def fn():
        for arr in layer_data.values():
            if arr is not None:
                _ = float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr))
        if state_256.fr_midget_on_L is not None:
            _ = float(np.mean(state_256.fr_midget_on_L))
            flat = state_256.fr_midget_on_L.flatten()
            flat = flat[np.isfinite(flat)]
            if len(flat) > 0:
                _ = np.histogram(flat, bins=50)

    mean_ms, std_ms = _timed(REPEAT, fn, "stats_numpy")
    print(f"  stats (numpy only): {mean_ms:7.2f} ± {std_ms:5.2f} ms")


def test_bench_cell_positions_cold_256():
    """Cell positions init_default (cold, once per grid): build all positions + trees."""
    state_256 = _state_256()
    cfg = state_256.config.retina
    n = state_256.grid_shape()[0]
    fovea = (n / 2.0, n / 2.0)

    def fn():
        cp = CellPositions(grid_size=n, microns_per_px=cfg.microns_per_px, fovea_center=fovea)
        cp.init_default(cone_subsample=1)

    mean_ms, std_ms = _timed(3, fn, "cell_positions_cold_256")
    print(f"  cell_positions cold (256): {mean_ms:7.2f} ± {std_ms:5.2f} ms  (once per grid)")

def test_bench_pick_and_connectivity_256():
    """Pick + connectivity (warm: trees already built)."""
    state_256 = _state_256()
    cfg = state_256.config.retina
    n = state_256.grid_shape()[0]
    fovea = (n / 2.0, n / 2.0)
    cp = CellPositions(grid_size=n, microns_per_px=cfg.microns_per_px, fovea_center=fovea)
    cp.init_default(cone_subsample=1)

    def fn():
        pick_layer, cell_id = pick_nearest_cell_any_layer(cp, n / 2, n / 2, 20.0)
        if pick_layer == "RGC" and cell_id is not None:
            compute_rgc_connectivity(cp, cell_id, fovea, firing_rate=60.0)
        elif pick_layer == "Cone" and cell_id is not None:
            compute_cone_connectivity(cp, cell_id, fovea)

    mean_ms, std_ms = _timed(REPEAT, fn, "pick_connectivity_256")
    print(f"  pick + connectivity(256): {mean_ms:7.2f} ± {std_ms:5.2f} ms  (per click)")


def test_bench_full_frame_estimate_256():
    """Rough 'one main-thread frame' (no DPG): state read + 2D render + stats numpy."""
    state_256 = _state_256()
    layer = state_256.fr_midget_on_L if state_256.fr_midget_on_L is not None else np.zeros(state_256.grid_shape(), dtype=np.float32)
    DISPLAY_SCALE = 4
    MAX_DISPLAY_SIDE = 1024
    layer_data = {
        "RGC": state_256.fr_midget_on_L,
        "Cones L": state_256.cone_L,
    }

    def one_frame():
        rgba = grid_to_rgba(layer, colormap="firing")
        if rgba.shape[0] > MAX_DISPLAY_SIDE or rgba.shape[1] > MAX_DISPLAY_SIDE:
            rgba = block_average_downsample_rgba(rgba, MAX_DISPLAY_SIDE)
        else:
            rgba = np.repeat(np.repeat(rgba, DISPLAY_SCALE, axis=0), DISPLAY_SCALE, axis=1)
        _ = np.ascontiguousarray(rgba.astype(np.float32)).flatten()
        for arr in layer_data.values():
            if arr is not None:
                _ = float(np.mean(arr)), float(np.std(arr))

    mean_ms, std_ms = _timed(REPEAT, one_frame, "frame_256")
    print(f"  full frame estimate (256, no DPG): {mean_ms:7.2f} ± {std_ms:5.2f} ms  (target < {FRAME_BUDGET_MS:.0f} ms)")
    print(f"  -> estimated max FPS (256): {1000.0 / mean_ms:.0f}")


def test_bench_summary():
    """Print a one-line summary of where time goes (run last with -s)."""
    state_256 = _state_256()
    results = []
    # Tick
    mean, _ = _timed(5, lambda: tick(state_256, 1.0 / 60.0), "tick")
    results.append(("tick(256)", mean))
    # Render
    layer = state_256.fr_midget_on_L if state_256.fr_midget_on_L is not None else np.zeros(state_256.grid_shape(), dtype=np.float32)
    mean, _ = _timed(10, lambda: grid_to_rgba(layer, "firing"), "grid_to_rgba")
    results.append(("grid_to_rgba(256)", mean))
    mean, _ = _timed(10, lambda: np.repeat(np.repeat(grid_to_rgba(layer, "firing"), 4, axis=0), 4, axis=1), "upscale")
    results.append(("rgba+upscale(256)", mean))
    print("\n  --- Summary (ms) ---")
    for name, ms in results:
        bar = "!" if ms > FRAME_BUDGET_MS else " "
        print(f"  {bar} {name}: {ms:6.2f} ms")
    total = sum(r[1] for r in results)
    print(f"  (tick runs in worker; main thread ~ grid_to_rgba + upscale + DPG)")


if __name__ == "__main__":
    """Run benchmarks and print report without pytest: python tests/bench_performance.py"""
    cfg = default_config()
    state_256 = SimState(config=cfg)
    state_256.stimulus_params.update({"type": "spot", "intensity": 1.0, "x_deg": 0.0, "y_deg": 0.0})
    state_256.ensure_initialized()

    print("=== Performance benchmarks (target 60 FPS = 16.67 ms/frame) ===\n")

    # Tick
    mean, std = _timed(REPEAT, lambda: tick(state_256, 1.0 / 60.0), "tick")
    print(f"  tick(256):           {mean:7.2f} ± {std:5.2f} ms  {'OK' if mean < FRAME_BUDGET_MS else 'SLOW'}")
    # Diagnose whether Cython conv is used (must run from repo root so hot_numerical is on path)
    try:
        from hot_numerical.convolve_2d import gaussian_pool_2d as _cy_conv
        _arr = np.zeros((256, 256), dtype=np.float32)
        _t0 = time.perf_counter()
        for _ in range(11):
            _cy_conv(_arr, 12.0, "reflect")
        _el = (time.perf_counter() - _t0) * 1000 / 11
        print(f"  Cython convolve_2d:   loaded  (1× conv ~{_el:.2f} ms)")
    except Exception as e:
        print(f"  Cython convolve_2d:   NOT loaded  ({e})")
        print("      Run from repo root: python tests/bench_performance.py   (so hot_numerical is on path)")
    if mean >= FRAME_BUDGET_MS:
        print("      -> Build hot_numerical (cd hot_numerical && python setup.py build_ext --inplace) for 60 Hz.")
    # Tick breakdown: stimulus vs 11× convolution (main cost)
    cfg = state_256.config
    params = state_256.stimulus_params
    gh, gw = state_256.grid_shape()
    mean_s, _ = _timed(REPEAT, lambda: build_stimulus_spectrum(params, cfg.spectral, (gh, gw), 0.0, cfg.retina), "stimulus")
    arr = np.zeros((gh, gw), dtype=np.float32)
    sigma = 12.0  # typical for 256 grid
    def eleven_conv():
        for _ in range(11):
            gaussian_pool_2d(arr, sigma, mode="reflect")
    mean_c, _ = _timed(3, eleven_conv, "11×gaussian_pool_2d")
    mean_c /= 11.0
    print(f"  -> stimulus(spot):   {mean_s:7.2f} ms")
    print(f"  -> 1× gaussian_2d:   {mean_c:7.2f} ms  (×11 ≈ {11*mean_c:.0f} ms in tick)")
    # Render 2D
    layer = state_256.fr_midget_on_L if state_256.fr_midget_on_L is not None else np.zeros(state_256.grid_shape(), dtype=np.float32)
    mean, std = _timed(REPEAT, lambda: grid_to_rgba(layer, "firing"), "grid_to_rgba")
    print(f"  grid_to_rgba(256):   {mean:7.2f} ± {std:5.2f} ms")
    rgba = grid_to_rgba(layer, "firing")
    mean, std = _timed(REPEAT, lambda: np.repeat(np.repeat(rgba, 4, axis=0), 4, axis=1), "upscale")
    print(f"  upscale 256->1024:   {mean:7.2f} ± {std:5.2f} ms")
    # Full frame estimate
    def one_frame():
        r = grid_to_rgba(layer, "firing")
        r = np.repeat(np.repeat(r, 4, axis=0), 4, axis=1)
        _ = np.ascontiguousarray(r.astype(np.float32)).flatten()
    mean, std = _timed(REPEAT, one_frame, "frame")
    print(f"  frame (256, no DPG): {mean:7.2f} ± {std:5.2f} ms  -> ~{1000/max(0.01, mean):.0f} FPS")
    # Pick + connectivity
    n = state_256.grid_shape()[0]
    fovea = (n / 2.0, n / 2.0)
    cp = CellPositions(grid_size=n, microns_per_px=state_256.config.retina.microns_per_px, fovea_center=fovea)
    cp.init_default(cone_subsample=1)
    def pick_conn():
        pl, cid = pick_nearest_cell_any_layer(cp, n/2, n/2, 20.0)
        if pl == "RGC" and cid is not None:
            compute_rgc_connectivity(cp, cid, fovea, firing_rate=60.0)
        elif pl == "Cone" and cid is not None:
            compute_cone_connectivity(cp, cid, fovea)
    mean, std = _timed(REPEAT, pick_conn, "pick_conn")
    print(f"  pick + connectivity:  {mean:7.2f} ± {std:5.2f} ms  (per click)")
    # 2048 tick (optional; set BENCH_SKIP_2048=1 to skip — 2048 tick can take 1+ min)
    if not os.environ.get("BENCH_SKIP_2048", "").strip().lower() in ("1", "true", "yes"):
        try:
            cfg_big = large_field_config()
            state_2048 = SimState(config=cfg_big)
            state_2048.stimulus_params.update({"type": "spot", "intensity": 1.0})
            state_2048.ensure_initialized()
            mean, std = _timed(1, lambda: tick(state_2048, 1.0/60.0), "tick_2048")  # 1 run (slow)
            print(f"  tick(2048):          {mean:7.2f} ± {std:5.2f} ms  (large field)")
            layer_big = state_2048.fr_midget_on_L if state_2048.fr_midget_on_L is not None else np.zeros(state_2048.grid_shape(), dtype=np.float32)
            mean, std = _timed(3, lambda: block_average_downsample_rgba(grid_to_rgba(layer_big, "firing"), 1024), "render_2048")
            print(f"  render_2d(2048):    {mean:7.2f} ± {std:5.2f} ms")
        except Exception as e:
            print(f"  (2048 benchmarks skipped: {e})")
    else:
        print("  (2048 skipped; unset BENCH_SKIP_2048 to run)")
    print("\nDone.")
