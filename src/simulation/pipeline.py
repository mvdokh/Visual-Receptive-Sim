from __future__ import annotations

"""
Vectorized simulation pipeline for the retinal circuit.

This follows the high-level steps described in the project spec:

1. Build stimulus spectrum grid
2. Cone responses (L, M, S)
3. Horizontal pooling
4. Horizontal → cone feedback (surround)
5. Bipolar responses
6. Amacrine lateral inhibition
7. RGC generators (dendritic integration)
8. LN nonlinearity → firing rate
9. Color opponent signals
10. Temporal smoothing
11. Mark textures dirty
"""

from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter

from src.config import GlobalConfig
from src.simulation.state import SimState
from src.simulation.stimulus.spectral import build_stimulus_spectrum


SMOOTHED_LAYERS: Iterable[str] = [
    "cone_L",
    "cone_M",
    "cone_S",
    "h_activation",
    "bp_midget_on_L",
    "bp_midget_off_L",
    "bp_midget_on_M",
    "bp_midget_off_M",
    "bp_diffuse_on",
    "bp_diffuse_off",
    "amacrine_aii",
    "amacrine_wide",
    "rgc_midget_on_L",
    "rgc_midget_off_L",
    "rgc_midget_on_M",
    "rgc_midget_off_M",
    "rgc_parasol_on",
    "rgc_parasol_off",
    "fr_midget_on_L",
    "fr_midget_off_L",
    "fr_midget_on_M",
    "fr_midget_off_M",
    "fr_parasol_on",
    "fr_parasol_off",
]


def _sigmoid(x: np.ndarray, r_max: float, x_half: float, slope: float) -> np.ndarray:
    return r_max / (1.0 + np.exp(-slope * (x - x_half)))


def tick(state: SimState, dt: float) -> None:
    """
    Advance the simulation by one time step of length `dt` (seconds).

    All operations are fully vectorized over the retinal grid.
    """
    state.ensure_initialized()
    cfg: GlobalConfig = state.config
    state.time += dt

    # 1. Stimulus spectrum grid (H, W, L)
    state.stimulus_spectrum = build_stimulus_spectrum(
        state.stimulus_params, cfg.spectral, state.grid_shape(), time_s=state.time
    )

    # 2. Cone responses via spectral dot product
    stim = state.stimulus_spectrum  # (H, W, L)
    SENS_L = cfg.spectral.sens_L.astype(np.float32)
    SENS_M = cfg.spectral.sens_M.astype(np.float32)
    SENS_S = cfg.spectral.sens_S.astype(np.float32)

    state.cone_L = np.einsum("hwl,l->hw", stim, SENS_L, optimize=True)
    state.cone_M = np.einsum("hwl,l->hw", stim, SENS_M, optimize=True)
    state.cone_S = np.einsum("hwl,l->hw", stim, SENS_S, optimize=True)

    # 3. Horizontal cell pooling (cone_to_horizontal scales cone input)
    cw = getattr(cfg, "connectivity_weights", None)
    cone_to_h = cw.cone_to_horizontal if cw else 1.0
    sigma_H = cfg.horizontal.sigma_lm_deg / cfg.retina.dx_deg
    sigma_H_s = cfg.horizontal.sigma_s_deg / cfg.retina.dx_deg
    cone_lm = (state.cone_L + state.cone_M) * cone_to_h
    cone_s_in = state.cone_S * cone_to_h

    h_lm = gaussian_filter(cone_lm, sigma=sigma_H, mode="reflect")
    h_s = gaussian_filter(cone_s_in, sigma=sigma_H_s, mode="reflect")
    h_to_cone = cw.horizontal_to_cone if cw else 1.0
    state.h_activation = (
        h_lm * cfg.horizontal.alpha_lm * h_to_cone + h_s * cfg.horizontal.alpha_s * h_to_cone
    )

    # 4. Horizontal → cone feedback (surround)
    state.cone_L_eff = state.cone_L - cfg.horizontal.alpha_lm * state.h_activation
    state.cone_M_eff = state.cone_M - cfg.horizontal.alpha_lm * state.h_activation
    state.cone_S_eff = state.cone_S - cfg.horizontal.alpha_s * state.h_activation

    # 5. Bipolar responses (cone_to_bipolar scales effective cone input)
    cone_to_bp = cw.cone_to_bipolar if cw else 1.0
    sigma_diffuse = cfg.bipolar.sigma_diffuse_deg / cfg.retina.dx_deg
    cone_lm_eff = (state.cone_L_eff + state.cone_M_eff) * cone_to_bp

    state.bp_midget_on_L = np.maximum(0.0, state.cone_L_eff * cone_to_bp)
    state.bp_midget_off_L = np.maximum(0.0, -state.cone_L_eff * cone_to_bp)
    state.bp_midget_on_M = np.maximum(0.0, state.cone_M_eff * cone_to_bp)
    state.bp_midget_off_M = np.maximum(0.0, -state.cone_M_eff * cone_to_bp)

    pooled = gaussian_filter(cone_lm_eff, sigma=sigma_diffuse, mode="reflect")
    state.bp_diffuse_on = np.maximum(0.0, pooled)
    state.bp_diffuse_off = np.maximum(0.0, -pooled)

    # 6. Amacrine lateral inhibition (bipolar_to_amacrine, amacrine_to_bipolar)
    bp_to_am = cw.bipolar_to_amacrine if cw else 1.0
    am_to_bp = cw.amacrine_to_bipolar if cw else 1.0
    sigma_aii = cfg.amacrine.sigma_aii_deg / cfg.retina.dx_deg
    sigma_wide = cfg.amacrine.sigma_wide_deg / cfg.retina.dx_deg

    state.amacrine_aii = gaussian_filter(
        (state.bp_midget_on_L + state.bp_midget_on_M) * bp_to_am, sigma=sigma_aii, mode="reflect"
    )
    state.amacrine_wide = gaussian_filter(cone_lm_eff * bp_to_am, sigma=sigma_wide, mode="reflect")

    total_amacrine = (
        cfg.amacrine.gamma_aii * am_to_bp * state.amacrine_aii
        + cfg.amacrine.gamma_wide * am_to_bp * state.amacrine_wide
    )

    # 7. RGC generators (bipolar_to_rgc scales drive)
    bp_to_rgc = cw.bipolar_to_rgc if cw else 1.0

    def rgc_generator(bp_grid: np.ndarray, sigma_deg: float) -> np.ndarray:
        sigma_px = sigma_deg / cfg.retina.dx_deg
        drive = (bp_grid - total_amacrine) * bp_to_rgc
        return gaussian_filter(drive, sigma=sigma_px, mode="reflect")

    state.rgc_midget_on_L = rgc_generator(
        state.bp_midget_on_L, cfg.dendritic.sigma_midget_deg
    )
    state.rgc_midget_off_L = rgc_generator(
        state.bp_midget_off_L, cfg.dendritic.sigma_midget_deg
    )
    state.rgc_midget_on_M = rgc_generator(
        state.bp_midget_on_M, cfg.dendritic.sigma_midget_deg
    )
    state.rgc_midget_off_M = rgc_generator(
        state.bp_midget_off_M, cfg.dendritic.sigma_midget_deg
    )
    state.rgc_parasol_on = rgc_generator(
        state.bp_diffuse_on, cfg.dendritic.sigma_parasol_deg
    )
    state.rgc_parasol_off = rgc_generator(
        state.bp_diffuse_off, cfg.dendritic.sigma_parasol_deg
    )

    # 8. LN sigmoid → firing rates
    nl = cfg.rgc_nl
    state.fr_midget_on_L = _sigmoid(state.rgc_midget_on_L, nl.r_max, nl.x_half, nl.slope)
    state.fr_midget_off_L = _sigmoid(
        state.rgc_midget_off_L, nl.r_max, nl.x_half, nl.slope
    )
    state.fr_midget_on_M = _sigmoid(state.rgc_midget_on_M, nl.r_max, nl.x_half, nl.slope)
    state.fr_midget_off_M = _sigmoid(
        state.rgc_midget_off_M, nl.r_max, nl.x_half, nl.slope
    )
    state.fr_parasol_on = _sigmoid(state.rgc_parasol_on, nl.r_max, nl.x_half, nl.slope)
    state.fr_parasol_off = _sigmoid(
        state.rgc_parasol_off, nl.r_max, nl.x_half, nl.slope
    )

    # 9. Color opponent signals
    state.lm_opponent = state.fr_midget_on_L - state.fr_midget_on_M
    state.by_opponent = state.cone_S_eff - 0.5 * (
        state.cone_L_eff + state.cone_M_eff
    )

    # 10. Temporal smoothing (per-layer RC filter)
    tau_map = {
        "cone_L": cfg.temporal.cone_tau,
        "cone_M": cfg.temporal.cone_tau,
        "cone_S": cfg.temporal.cone_tau,
        "h_activation": cfg.temporal.horizontal_tau,
        "bp_midget_on_L": cfg.temporal.bipolar_tau,
        "bp_midget_off_L": cfg.temporal.bipolar_tau,
        "bp_midget_on_M": cfg.temporal.bipolar_tau,
        "bp_midget_off_M": cfg.temporal.bipolar_tau,
        "bp_diffuse_on": cfg.temporal.bipolar_tau,
        "bp_diffuse_off": cfg.temporal.bipolar_tau,
        "amacrine_aii": cfg.temporal.amacrine_tau,
        "amacrine_wide": cfg.temporal.amacrine_tau,
        "rgc_midget_on_L": cfg.temporal.rgc_tau,
        "rgc_midget_off_L": cfg.temporal.rgc_tau,
        "rgc_midget_on_M": cfg.temporal.rgc_tau,
        "rgc_midget_off_M": cfg.temporal.rgc_tau,
        "rgc_parasol_on": cfg.temporal.rgc_tau,
        "rgc_parasol_off": cfg.temporal.rgc_tau,
        "fr_midget_on_L": cfg.temporal.rgc_tau,
        "fr_midget_off_L": cfg.temporal.rgc_tau,
        "fr_midget_on_M": cfg.temporal.rgc_tau,
        "fr_midget_off_M": cfg.temporal.rgc_tau,
        "fr_parasol_on": cfg.temporal.rgc_tau,
        "fr_parasol_off": cfg.temporal.rgc_tau,
    }
    for attr in SMOOTHED_LAYERS:
        tau = tau_map.get(attr, cfg.temporal.rgc_tau)
        alpha = float(dt / max(tau, 1e-6))
        alpha = max(0.0, min(alpha, 1.0))
        prev = state.smoothed[attr]
        curr = getattr(state, attr)
        smoothed = prev + (curr - prev) * alpha
        state.smoothed[attr] = smoothed
        setattr(state, attr, smoothed)

    # 11. Mark all textures dirty
    for key in state.dirty_flags:
        state.dirty_flags[key] = True

