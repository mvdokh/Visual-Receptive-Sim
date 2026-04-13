#!/usr/bin/env python3
"""
Time major simulator subsystems to spot Cython / NumPy bottlenecks.

Run from repository root (use the project venv if system Python lacks SciPy)::

    .venv/bin/python cython/benchmark_sim_components.py
    .venv/bin/python cython/benchmark_sim_components.py --grid 512 --repeats 25
    .venv/bin/python cython/benchmark_sim_components.py --profile-ticks 8 --no-micro

Uses ``time.perf_counter`` for coarse timings and ``cProfile`` for per-function
hot spots inside ``tick``. Prioritize Cython on functions with high *tottime*
that are pure array math in ``src/simulation`` or ``hot_numerical``.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import GlobalConfig, SpatialHeterogeneityMode, default_config, large_field_config
from src.rendering.heatmap import block_average_downsample_rgba, grid_to_rgba
from src.simulation import SimState, tick
from src.simulation.fast_conv import gaussian_pool_2d
from src.simulation.fast_layers import sigmoid_ln, temporal_rc
from src.simulation.spatial_heterogeneity_maps import rebuild_spatial_heterogeneity
from src.simulation.stimulus.spectral import build_stimulus_spectrum


def _timed(repeat: int, fn: Callable[[], None]) -> tuple[float, float]:
    samples: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(samples)), float(np.std(samples))


def _hot_numerical_status() -> tuple[bool, bool, bool]:
    """Return (gaussian_pool_cython, sigmoid_ln_cython, stimulus_fill_cython)."""
    conv = layer = stim = False
    try:
        from hot_numerical.convolve_2d import gaussian_pool_2d as _  # noqa: F401

        conv = True
    except ImportError:
        pass
    try:
        from hot_numerical.layer_update import sigmoid_ln as _  # noqa: F401

        layer = True
    except ImportError:
        pass
    try:
        from hot_numerical.stimulus_fill import fill_spot_or_full as _  # noqa: F401

        stim = True
    except ImportError:
        pass
    return conv, layer, stim


def _make_state(cfg: GlobalConfig, stimulus: dict) -> SimState:
    s = SimState(config=cfg)
    s.stimulus_params.update(stimulus)
    s.ensure_initialized()
    return s


def run_micro_benchmarks(state: SimState, repeats: int) -> None:
    cfg = state.config
    dt = 1.0 / 60.0
    H, W = state.grid_shape()
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((H, W), dtype=np.float32).astype(np.float32)

    print("\n--- Micro-benchmarks (mean ± std ms) ---")

    mean_ms, std_ms = _timed(repeats, lambda: tick(state, dt))
    print(f"  tick (full pipeline)     {mean_ms:7.2f} ± {std_ms:5.2f}  ({repeats} runs)")

    # Heterogeneity rebuild (one-shot cost when maps are dirty)
    state_hetero = SimState(config=cfg)
    state_hetero.stimulus_params.update(state.stimulus_params)
    state_hetero.ensure_initialized()
    state_hetero.heterogeneity_dirty = True

    def _rebuild_once() -> None:
        state_hetero.heterogeneity_dirty = True
        rebuild_spatial_heterogeneity(state_hetero, cfg)
        state_hetero.heterogeneity_dirty = False

    mean_ms, std_ms = _timed(min(repeats, 8), _rebuild_once)
    print(f"  rebuild_spatial_hetero    {mean_ms:7.2f} ± {std_ms:5.2f}  (forced dirty)")

    # Stimulus spectrum only (same API as pipeline)
    mean_ms, std_ms = _timed(
        repeats,
        lambda: build_stimulus_spectrum(
            state.stimulus_params,
            cfg.spectral,
            state.grid_shape(),
            time_s=state.time,
            retina=cfg.retina,
        ),
    )
    print(f"  build_stimulus_spectrum  {mean_ms:7.2f} ± {std_ms:5.2f}")

    stim = state.stimulus_spectrum
    if stim is None:
        tick(state, dt)
        stim = state.stimulus_spectrum
    assert stim is not None
    SENS = cfg.spectral.sens_L.astype(np.float32)

    def _cone_einsum() -> None:
        np.einsum("hwl,l->hw", stim, SENS, optimize=True)

    mean_ms, std_ms = _timed(repeats, _cone_einsum)
    print(f"  einsum cone_L (1 of 3)   {mean_ms:7.2f} ± {std_ms:5.2f}")

    sigma_h = cfg.horizontal.sigma_lm_deg / cfg.retina.dx_deg
    sigma_h = float(max(sigma_h, 0.5))

    def _one_pool() -> None:
        gaussian_pool_2d(noise, sigma_h, mode="reflect")

    mean_ms, std_ms = _timed(repeats, _one_pool)
    print(f"  gaussian_pool_2d (1×)    {mean_ms:7.2f} ± {std_ms:5.2f}  sigma_px={sigma_h:.2f}")

    n_pools = 18

    def _many_pools() -> None:
        for _ in range(n_pools):
            gaussian_pool_2d(noise, sigma_h, mode="reflect")

    mean_ms, std_ms = _timed(max(3, repeats // 3), _many_pools)
    per = mean_ms / n_pools
    print(f"  gaussian_pool_2d ({n_pools}×) {mean_ms:7.2f} ± {std_ms:5.2f}  (~{per:.3f} ms / pool)")

    nl = cfg.rgc_nl
    rgc_like = noise * 0.5 + 0.1

    def _six_sigmoid() -> None:
        for _ in range(6):
            sigmoid_ln(rgc_like, nl.r_max, nl.x_half, nl.slope)

    mean_ms, std_ms = _timed(repeats, _six_sigmoid)
    print(f"  sigmoid_ln (6× FR-like)  {mean_ms:7.2f} ± {std_ms:5.2f}")

    prev = rgc_like.copy()
    alpha = 0.1

    def _many_temporal() -> None:
        for _ in range(24):
            temporal_rc(prev, rgc_like, alpha)

    mean_ms, std_ms = _timed(repeats, _many_temporal)
    print(f"  temporal_rc (24×)       {mean_ms:7.2f} ± {std_ms:5.2f}")

    layer = state.fr_midget_on_L
    if layer is None:
        tick(state, dt)
        layer = state.fr_midget_on_L
    assert layer is not None
    DISPLAY_SCALE = 4
    MAX_DISPLAY_SIDE = 1024

    def _render_frame() -> None:
        rgba = grid_to_rgba(layer, colormap="firing")
        if rgba.shape[0] > MAX_DISPLAY_SIDE or rgba.shape[1] > MAX_DISPLAY_SIDE:
            rgba = block_average_downsample_rgba(rgba, MAX_DISPLAY_SIDE)
        else:
            rgba = np.repeat(np.repeat(rgba, DISPLAY_SCALE, axis=0), DISPLAY_SCALE, axis=1)
        _ = np.ascontiguousarray(rgba.astype(np.float32)).ravel()

    mean_ms, std_ms = _timed(repeats, _render_frame)
    print(f"  render path (heatmap)    {mean_ms:7.2f} ± {std_ms:5.2f}  (grid_to_rgba + scale)")

    # Optional: heterogeneous mode cost vs homogeneous (same grid)
    if H <= 512:
        sc_cfg = default_config()
        sc_cfg.retina.grid_resolution = H
        sc_cfg.retina.field_size_deg = cfg.retina.field_size_deg
        sc_cfg.spatial_heterogeneity.mode = SpatialHeterogeneityMode.SCATTER
        sc_cfg.spatial_heterogeneity.scatter.sigma = 0.25
        s2 = _make_state(
            sc_cfg,
            {"type": "spot", "intensity": 1.0, "x_deg": 0.0, "y_deg": 0.0},
        )
        mean_ms, std_ms = _timed(max(5, repeats // 2), lambda: tick(s2, dt))
        print(f"  tick (SCATTER mode)      {mean_ms:7.2f} ± {std_ms:5.2f}")


def run_profile(state: SimState, n_ticks: int, top_n: int) -> None:
    dt = 1.0 / 60.0
    pr = cProfile.Profile()

    def _work() -> None:
        for _ in range(n_ticks):
            tick(state, dt)

    pr.enable()
    _work()
    pr.disable()

    stream = io.StringIO()
    stats = pstats.Stats(pr, stream=stream)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats(top_n)
    print(f"\n--- cProfile: top {top_n} by tottime over {n_ticks} tick(s) ---")
    print(stream.getvalue())

    stream2 = io.StringIO()
    stats2 = pstats.Stats(pr, stream=stream2)
    stats2.sort_stats(pstats.SortKey.CUMULATIVE)
    stats2.print_stats(top_n)
    print(f"\n--- cProfile: top {top_n} by cumulative time ---")
    print(stream2.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark simulator components for Cython targets.")
    parser.add_argument(
        "--grid",
        type=int,
        default=256,
        help="Grid resolution per side (default 256; use 512, 1024, … if config allows).",
    )
    parser.add_argument("--repeats", type=int, default=20, help="Micro-benchmark repetitions.")
    parser.add_argument("--profile-ticks", type=int, default=6, help="Ticks to profile (0 = skip cProfile).")
    parser.add_argument("--profile-top", type=int, default=35, help="Lines to print from cProfile.")
    parser.add_argument("--large-field", action="store_true", help="Use large_field_config() (e.g. 2048²).")
    parser.add_argument("--no-micro", action="store_true", help="Only run cProfile section.")
    args = parser.parse_args()

    os.environ.setdefault("PYTHONWARNINGS", "ignore")

    conv_cy, layer_cy, stim_cy = _hot_numerical_status()
    print("hot_numerical extensions:")
    print(f"  convolve_2d.gaussian_pool_2d  {'yes' if conv_cy else 'no (SciPy path)'}")
    print(f"  layer_update.sigmoid_ln       {'yes' if layer_cy else 'no (NumPy path)'}")
    print(f"  stimulus_fill.fill_spot_or_full {'yes' if stim_cy else 'no / not built'}")
    print(f"  HOT_NUMERICAL_USE_CYTHON_CONV={os.environ.get('HOT_NUMERICAL_USE_CYTHON_CONV', '1')}")

    if args.large_field:
        cfg = large_field_config()
    else:
        cfg = default_config()
        cfg.retina.grid_resolution = int(args.grid)

    stimulus = {"type": "spot", "intensity": 1.0, "x_deg": 0.0, "y_deg": 0.0}
    state = _make_state(cfg, stimulus)
    # Warm JIT / caches
    for _ in range(3):
        tick(state, 1.0 / 60.0)

    print(f"\nConfig: grid {state.grid_shape()[0]}×{state.grid_shape()[1]}, field_deg={cfg.retina.field_size_deg}")

    if not args.no_micro:
        run_micro_benchmarks(state, args.repeats)

    if args.profile_ticks > 0:
        run_profile(state, args.profile_ticks, args.profile_top)

    print(
        "\nInterpretation: small-σ pools use Cython; σ above "
        "~SIGMA_FFT_THRESHOLD_PX use FFT Gaussian (see fast_conv). "
        "Heavy hitters are still many gaussian_pool_2d calls per tick — "
        "next wins are fusing passes in the pipeline or extending a tuned "
        "Cython separable path for larger σ."
    )


if __name__ == "__main__":
    main()
