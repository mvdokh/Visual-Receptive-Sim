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

from src.config import GlobalConfig
from src.simulation.rgc_population import (
    calibrated_dendritic_sigmas_deg,
    compute_cross_type_rf_modulation,
    population_fractions_from_config,
)
from src.simulation.state import SimState
from src.simulation.stimulus.spectral import build_stimulus_spectrum
from src.simulation.fast_conv import gaussian_pool_2d
from src.simulation.fast_layers import sigmoid_ln, temporal_rc


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
def tick(state: SimState, dt: float) -> None:
    """
    Advance the simulation by one time step of length `dt` (seconds).

    All operations are fully vectorized over the retinal grid.
    """
    state.ensure_initialized()
    cfg: GlobalConfig = state.config
    state.time += dt

    rpc = getattr(cfg, "rgc_population", None)
    if rpc is not None and rpc.enabled:
        tf_rpc = population_fractions_from_config(rpc)
        mod = compute_cross_type_rf_modulation(tf_rpc)
        alpha_lm_use = cfg.horizontal.alpha_lm * mod["horizontal_alpha_lm_scale"]
        alpha_s_use = cfg.horizontal.alpha_s * mod["horizontal_alpha_lm_scale"]
        gm_aii = cfg.amacrine.gamma_aii * mod["gamma_aii_scale"]
        gm_wide = cfg.amacrine.gamma_wide * mod["gamma_wide_scale"]
        sm_deg, sp_deg, si_m_rel, si_p_rel, r_scale, sur_m_deg, sur_p_deg = (
            calibrated_dendritic_sigmas_deg(
                tf_rpc,
                cfg.dendritic.sigma_midget_deg,
                cfg.dendritic.sigma_parasol_deg,
                rpc.t5_cluster_bias,
            )
        )
        r_max_eff = cfg.rgc_nl.r_max * r_scale
    else:
        alpha_lm_use = cfg.horizontal.alpha_lm
        alpha_s_use = cfg.horizontal.alpha_s
        gm_aii = cfg.amacrine.gamma_aii
        gm_wide = cfg.amacrine.gamma_wide
        sm_deg = cfg.dendritic.sigma_midget_deg
        sp_deg = cfg.dendritic.sigma_parasol_deg
        si_m_rel = 0.0
        si_p_rel = 0.0
        sur_m_deg = 0.0
        sur_p_deg = 0.0
        r_max_eff = cfg.rgc_nl.r_max

    # 1. Stimulus spectrum grid (H, W, L); pass retina so 1° scales with grid
    state.stimulus_spectrum = build_stimulus_spectrum(
        state.stimulus_params,
        cfg.spectral,
        state.grid_shape(),
        time_s=state.time,
        retina=cfg.retina,
    )

    # 2. Cone responses via spectral dot product, then Naka-Rushton saturation
    stim = state.stimulus_spectrum  # (H, W, L)
    SENS_L = cfg.spectral.sens_L.astype(np.float32)
    SENS_M = cfg.spectral.sens_M.astype(np.float32)
    SENS_S = cfg.spectral.sens_S.astype(np.float32)

    linear_L = np.einsum("hwl,l->hw", stim, SENS_L, optimize=True)
    linear_M = np.einsum("hwl,l->hw", stim, SENS_M, optimize=True)
    linear_S = np.einsum("hwl,l->hw", stim, SENS_S, optimize=True)

    sigma = getattr(cfg.spectral, "cone_saturation_sigma", 0.0)
    if sigma > 0:
        # Naka-Rushton: R = I / (I + sigma) so cone response scales with intensity and saturates.
        state.cone_L = np.clip(linear_L / (linear_L + sigma), 0.0, 1.0).astype(np.float32)
        state.cone_M = np.clip(linear_M / (linear_M + sigma), 0.0, 1.0).astype(np.float32)
        state.cone_S = np.clip(linear_S / (linear_S + sigma), 0.0, 1.0).astype(np.float32)
    else:
        state.cone_L = linear_L.astype(np.float32)
        state.cone_M = linear_M.astype(np.float32)
        state.cone_S = linear_S.astype(np.float32)

    # 3. Horizontal cell pooling (cone_to_horizontal scales cone input)
    cw = getattr(cfg, "connectivity_weights", None)
    cone_to_h = cw.cone_to_horizontal if cw else 1.0
    sigma_H = cfg.horizontal.sigma_lm_deg / cfg.retina.dx_deg
    sigma_H_s = cfg.horizontal.sigma_s_deg / cfg.retina.dx_deg
    cone_lm = (state.cone_L + state.cone_M) * cone_to_h
    cone_s_in = state.cone_S * cone_to_h

    h_lm = gaussian_pool_2d(cone_lm, sigma_H, mode="reflect")
    h_s = gaussian_pool_2d(cone_s_in, sigma_H_s, mode="reflect")
    h_to_cone = cw.horizontal_to_cone if cw else 1.0
    state.h_activation = (
        h_lm * alpha_lm_use * h_to_cone + h_s * alpha_s_use * h_to_cone
    )

    # 4. Horizontal → cone feedback (surround). Rectify so center is not over-suppressed
    # (avoids a bright ring at the spot edge; cone output is non-negative in standard models).
    state.cone_L_eff = np.maximum(
        0.0,
        state.cone_L - alpha_lm_use * state.h_activation,
    ).astype(np.float32)
    state.cone_M_eff = np.maximum(
        0.0,
        state.cone_M - alpha_lm_use * state.h_activation,
    ).astype(np.float32)
    state.cone_S_eff = np.maximum(
        0.0,
        state.cone_S - alpha_s_use * state.h_activation,
    ).astype(np.float32)

    # 5. Bipolar responses (cone_to_bipolar scales effective cone input)
    cone_to_bp = cw.cone_to_bipolar if cw else 1.0
    sigma_diffuse = cfg.bipolar.sigma_diffuse_deg / cfg.retina.dx_deg
    cone_lm_eff = (state.cone_L_eff + state.cone_M_eff) * cone_to_bp

    state.bp_midget_on_L = np.maximum(0.0, state.cone_L_eff * cone_to_bp)
    state.bp_midget_off_L = np.maximum(0.0, -state.cone_L_eff * cone_to_bp)
    state.bp_midget_on_M = np.maximum(0.0, state.cone_M_eff * cone_to_bp)
    state.bp_midget_off_M = np.maximum(0.0, -state.cone_M_eff * cone_to_bp)

    pooled = gaussian_pool_2d(cone_lm_eff, sigma_diffuse, mode="reflect")
    state.bp_diffuse_on = np.maximum(0.0, pooled)
    state.bp_diffuse_off = np.maximum(0.0, -pooled)

    # 6. Amacrine lateral inhibition (bipolar_to_amacrine, amacrine_to_bipolar)
    bp_to_am = cw.bipolar_to_amacrine if cw else 1.0
    am_to_bp = cw.amacrine_to_bipolar if cw else 1.0
    sigma_aii = cfg.amacrine.sigma_aii_deg / cfg.retina.dx_deg
    sigma_wide = cfg.amacrine.sigma_wide_deg / cfg.retina.dx_deg

    state.amacrine_aii = gaussian_pool_2d(
        (state.bp_midget_on_L + state.bp_midget_on_M) * bp_to_am, sigma_aii, mode="reflect"
    )
    state.amacrine_wide = gaussian_pool_2d(cone_lm_eff * bp_to_am, sigma_wide, mode="reflect")

    total_amacrine = (
        gm_aii * am_to_bp * state.amacrine_aii
        + gm_wide * am_to_bp * state.amacrine_wide
    )

    # 7. RGC generators (bipolar_to_rgc scales drive)
    bp_to_rgc = cw.bipolar_to_rgc if cw else 1.0

    def rgc_generator(
        bp_grid: np.ndarray,
        sigma_center_deg: float,
        sigma_surround_deg: float,
        si_rel: float,
    ) -> np.ndarray:
        sigma_c_px = sigma_center_deg / cfg.retina.dx_deg
        drive = (bp_grid - total_amacrine) * bp_to_rgc
        center = gaussian_pool_2d(drive, sigma_c_px, mode="reflect")
        if si_rel <= 1e-12 or sigma_surround_deg <= 1e-12:
            return center
        sigma_s_px = sigma_surround_deg / cfg.retina.dx_deg
        surr = gaussian_pool_2d(drive, sigma_s_px, mode="reflect")
        return center - si_rel * surr

    state.rgc_midget_on_L = rgc_generator(
        state.bp_midget_on_L, sm_deg, sur_m_deg, si_m_rel
    )
    state.rgc_midget_off_L = rgc_generator(
        state.bp_midget_off_L, sm_deg, sur_m_deg, si_m_rel
    )
    state.rgc_midget_on_M = rgc_generator(
        state.bp_midget_on_M, sm_deg, sur_m_deg, si_m_rel
    )
    state.rgc_midget_off_M = rgc_generator(
        state.bp_midget_off_M, sm_deg, sur_m_deg, si_m_rel
    )
    state.rgc_parasol_on = rgc_generator(
        state.bp_diffuse_on, sp_deg, sur_p_deg, si_p_rel
    )
    state.rgc_parasol_off = rgc_generator(
        state.bp_diffuse_off, sp_deg, sur_p_deg, si_p_rel
    )

    # 8. LN sigmoid → firing rates
    nl = cfg.rgc_nl
    state.fr_midget_on_L = sigmoid_ln(state.rgc_midget_on_L, r_max_eff, nl.x_half, nl.slope)
    state.fr_midget_off_L = sigmoid_ln(
        state.rgc_midget_off_L, r_max_eff, nl.x_half, nl.slope
    )
    state.fr_midget_on_M = sigmoid_ln(state.rgc_midget_on_M, r_max_eff, nl.x_half, nl.slope)
    state.fr_midget_off_M = sigmoid_ln(
        state.rgc_midget_off_M, r_max_eff, nl.x_half, nl.slope
    )
    state.fr_parasol_on = sigmoid_ln(state.rgc_parasol_on, r_max_eff, nl.x_half, nl.slope)
    state.fr_parasol_off = sigmoid_ln(
        state.rgc_parasol_off, r_max_eff, nl.x_half, nl.slope
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
        temporal_rc(prev, curr, alpha)
        # prev mutated in-place by temporal_rc
        state.smoothed[attr] = prev
        setattr(state, attr, prev)

    # 11. Mark all textures dirty
    for key in state.dirty_flags:
        state.dirty_flags[key] = True

