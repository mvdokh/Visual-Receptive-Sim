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

from src.config import GlobalConfig, SpatialHeterogeneityMode
from src.simulation.spatial_heterogeneity_maps import rebuild_spatial_heterogeneity
from src.simulation.state import SimState
from src.simulation.stimulus.spectral import build_stimulus_spectrum
from src.simulation.fast_conv import gaussian_pool_2d
from src.simulation.fast_layers import sigmoid_ln, temporal_rc

try:
    from hot_numerical.grid_ops import spectral_dot_hwl_l_to_hw as _spectral_dot_hwl_l_to_hw
except ImportError:  # pragma: no cover - optional Cython build
    _spectral_dot_hwl_l_to_hw = None


def _spectral_linear_response(stim: np.ndarray, sens: np.ndarray) -> np.ndarray:
    """(H,W,L) · sens -> (H,W); Cython when built, else NumPy einsum."""
    if _spectral_dot_hwl_l_to_hw is not None:
        return _spectral_dot_hwl_l_to_hw(
            np.ascontiguousarray(stim, dtype=np.float32),
            np.ascontiguousarray(sens, dtype=np.float32),
        )
    return np.einsum("hwl,l->hw", stim, sens, optimize=True)


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


def _gather_from_stack(stack: np.ndarray, idx_map: np.ndarray) -> np.ndarray:
    """stack (K, H, W); idx_map (H, W) with values in [0, K-1]."""
    ii = np.asarray(idx_map, dtype=np.int64)
    k = stack.shape[0]
    ii = np.clip(ii, 0, k - 1)
    return np.take_along_axis(stack, ii.reshape(1, *ii.shape), axis=0)[0]


def _rgc_pool_stack_bins(
    drive: np.ndarray,
    sigma_center_deg: np.ndarray,
    dx_deg: float,
) -> np.ndarray:
    K = int(sigma_center_deg.shape[0])
    out = np.empty((K,) + drive.shape, dtype=np.float32)
    for k in range(K):
        spx = float(sigma_center_deg[k]) / dx_deg
        spx = max(spx, 1e-6)
        out[k] = gaussian_pool_2d(drive, spx, mode="reflect")
    return out


def _rgc_pool_stack_type_map(
    drive: np.ndarray,
    gain: np.ndarray,
    sigma_center_deg: np.ndarray,
    dx_deg: float,
) -> np.ndarray:
    K = int(gain.shape[0])
    out = np.empty((K,) + drive.shape, dtype=np.float32)
    for k in range(K):
        dk = drive * float(gain[k])
        spx = float(sigma_center_deg[k]) / dx_deg
        spx = max(spx, 1e-6)
        out[k] = gaussian_pool_2d(dk, spx, mode="reflect")
    return out


def _mosaic_cell_mean(arr: np.ndarray, labels: np.ndarray, n_cell: int) -> np.ndarray:
    flat_a = arr.ravel().astype(np.float64)
    flat_l = labels.ravel().astype(np.int64)
    sums = np.bincount(flat_l, weights=flat_a, minlength=n_cell)
    cnt = np.bincount(flat_l, minlength=n_cell)
    return (sums / np.maximum(cnt, 1.0)).astype(np.float32)


def tick(state: SimState, dt: float) -> None:
    """
    Advance the simulation by one time step of length `dt` (seconds).

    All operations are fully vectorized over the retinal grid.
    """
    state.ensure_initialized()
    cfg: GlobalConfig = state.config

    if state.heterogeneity_dirty:
        rebuild_spatial_heterogeneity(state, cfg)
        state.heterogeneity_dirty = False

    state.time += dt

    hem = cfg.spatial_heterogeneity.mode

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
    dx_deg = cfg.retina.dx_deg

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

    linear_L = _spectral_linear_response(stim, SENS_L)
    linear_M = _spectral_linear_response(stim, SENS_M)
    linear_S = _spectral_linear_response(stim, SENS_S)

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
    sc_cb = (
        state.scatter_cone_to_bipolar
        if hem == SpatialHeterogeneityMode.SCATTER
        and state.scatter_cone_to_bipolar is not None
        else None
    )
    cb_m = sc_cb if sc_cb is not None else np.float32(1.0)
    sigma_diffuse = cfg.bipolar.sigma_diffuse_deg / cfg.retina.dx_deg
    cone_lm_eff = (state.cone_L_eff + state.cone_M_eff) * cone_to_bp * cb_m

    state.bp_midget_on_L = np.maximum(0.0, state.cone_L_eff * cone_to_bp * cb_m)
    state.bp_midget_off_L = np.maximum(0.0, -state.cone_L_eff * cone_to_bp * cb_m)
    state.bp_midget_on_M = np.maximum(0.0, state.cone_M_eff * cone_to_bp * cb_m)
    state.bp_midget_off_M = np.maximum(0.0, -state.cone_M_eff * cone_to_bp * cb_m)

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

    sc_am = (
        state.scatter_amacrine_to_bipolar
        if hem == SpatialHeterogeneityMode.SCATTER
        and state.scatter_amacrine_to_bipolar is not None
        else None
    )
    core_am = gm_aii * am_to_bp * state.amacrine_aii + gm_wide * am_to_bp * state.amacrine_wide
    total_amacrine = core_am * sc_am if sc_am is not None else core_am

    # 7. RGC generators (bipolar_to_rgc scales drive)
    bp_to_rgc = cw.bipolar_to_rgc if cw else 1.0
    sc_rgc = (
        state.scatter_bipolar_to_rgc
        if hem == SpatialHeterogeneityMode.SCATTER
        and state.scatter_bipolar_to_rgc is not None
        else None
    )
    rgc_m = sc_rgc if sc_rgc is not None else np.float32(1.0)

    def rgc_generator(
        bp_grid: np.ndarray,
        sigma_center_deg: float,
        sigma_surround_deg: float,
        si_rel: float,
    ) -> np.ndarray:
        sigma_c_px = sigma_center_deg / cfg.retina.dx_deg
        drive = (bp_grid - total_amacrine) * bp_to_rgc * rgc_m
        center = gaussian_pool_2d(drive, sigma_c_px, mode="reflect")
        if si_rel <= 1e-12 or sigma_surround_deg <= 1e-12:
            return center
        sigma_s_px = sigma_surround_deg / cfg.retina.dx_deg
        surr = gaussian_pool_2d(drive, sigma_s_px, mode="reflect")
        return center - si_rel * surr

    def drive_field(bp_grid: np.ndarray) -> np.ndarray:
        return (bp_grid - total_amacrine) * bp_to_rgc * rgc_m

    if hem == SpatialHeterogeneityMode.TYPE_MAP and state.type_map is not None:
        tm = state.type_map
        ptm = cfg.spatial_heterogeneity.type_map
        rf = np.asarray(ptm.rf_multiplier, dtype=np.float64)
        gn = np.asarray(ptm.gain_multiplier, dtype=np.float64)
        sigma_m = sm_deg * rf
        sigma_p = sp_deg * rf

        def rgc_tm(bp: np.ndarray, sig: np.ndarray) -> np.ndarray:
            d = drive_field(bp)
            stack = _rgc_pool_stack_type_map(d, gn.astype(np.float64), sig, dx_deg)
            return _gather_from_stack(stack, tm)

        state.rgc_midget_on_L = rgc_tm(state.bp_midget_on_L, sigma_m)
        state.rgc_midget_off_L = rgc_tm(state.bp_midget_off_L, sigma_m)
        state.rgc_midget_on_M = rgc_tm(state.bp_midget_on_M, sigma_m)
        state.rgc_midget_off_M = rgc_tm(state.bp_midget_off_M, sigma_m)
        state.rgc_parasol_on = rgc_tm(state.bp_diffuse_on, sigma_p)
        state.rgc_parasol_off = rgc_tm(state.bp_diffuse_off, sigma_p)
    elif (
        hem == SpatialHeterogeneityMode.ECCENTRICITY
        and state.eccentricity_bin_map is not None
        and state.eccentricity_bin_rep_scale is not None
    ):
        rep = np.asarray(state.eccentricity_bin_rep_scale, dtype=np.float64)
        bmap = state.eccentricity_bin_map
        sigma_m_bins = sm_deg * rep
        sigma_p_bins = sp_deg * rep

        def rgc_ecc(bp: np.ndarray, sig_bins: np.ndarray) -> np.ndarray:
            d = drive_field(bp)
            stack = _rgc_pool_stack_bins(d, sig_bins, dx_deg)
            return _gather_from_stack(stack, bmap)

        state.rgc_midget_on_L = rgc_ecc(state.bp_midget_on_L, sigma_m_bins)
        state.rgc_midget_off_L = rgc_ecc(state.bp_midget_off_L, sigma_m_bins)
        state.rgc_midget_on_M = rgc_ecc(state.bp_midget_on_M, sigma_m_bins)
        state.rgc_midget_off_M = rgc_ecc(state.bp_midget_off_M, sigma_m_bins)
        state.rgc_parasol_on = rgc_ecc(state.bp_diffuse_on, sigma_p_bins)
        state.rgc_parasol_off = rgc_ecc(state.bp_diffuse_off, sigma_p_bins)
    elif (
        hem == SpatialHeterogeneityMode.MOSAIC
        and state.voronoi_cell_id is not None
        and state.mosaic_n_cells > 0
    ):
        lab = state.voronoi_cell_id
        nc = int(state.mosaic_n_cells)

        def rgc_mosaic(bp: np.ndarray) -> np.ndarray:
            d = drive_field(bp)
            cell_m = _mosaic_cell_mean(d, lab, nc)
            return cell_m[lab]

        state.rgc_midget_on_L = rgc_mosaic(state.bp_midget_on_L)
        state.rgc_midget_off_L = rgc_mosaic(state.bp_midget_off_L)
        state.rgc_midget_on_M = rgc_mosaic(state.bp_midget_on_M)
        state.rgc_midget_off_M = rgc_mosaic(state.bp_midget_off_M)
        state.rgc_parasol_on = rgc_mosaic(state.bp_diffuse_on)
        state.rgc_parasol_off = rgc_mosaic(state.bp_diffuse_off)
    else:
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
    spo = getattr(cfg, "spike_output", None)
    spike_enabled = spo is not None and bool(spo.enabled)
    use_smooth_spikes = spike_enabled and bool(getattr(spo, "use_smoothed_rates", True))
    fr_ln_for_spikes: dict[str, np.ndarray] | None = None
    if spike_enabled and not use_smooth_spikes:
        fr_ln_for_spikes = {
            "fr_midget_on_L": np.asarray(state.fr_midget_on_L, dtype=np.float32).copy(),
            "fr_midget_off_L": np.asarray(state.fr_midget_off_L, dtype=np.float32).copy(),
            "fr_midget_on_M": np.asarray(state.fr_midget_on_M, dtype=np.float32).copy(),
            "fr_midget_off_M": np.asarray(state.fr_midget_off_M, dtype=np.float32).copy(),
            "fr_parasol_on": np.asarray(state.fr_parasol_on, dtype=np.float32).copy(),
            "fr_parasol_off": np.asarray(state.fr_parasol_off, dtype=np.float32).copy(),
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

    if spike_enabled:
        if state.spike_rng is None:
            state.spike_rng = np.random.default_rng(int(getattr(spo, "seed", 42)))
        rng = state.spike_rng

        def _bernoulli_spikes(rate_grid: np.ndarray) -> np.ndarray:
            r = np.maximum(np.asarray(rate_grid, dtype=np.float64), 0.0)
            p = 1.0 - np.exp(-r * float(dt))
            p = np.clip(p, 0.0, 1.0)
            u = rng.random(r.shape, dtype=np.float64)
            return (u < p).astype(np.float32)

        def _src(name: str) -> np.ndarray:
            if use_smooth_spikes:
                return getattr(state, name)
            assert fr_ln_for_spikes is not None
            return fr_ln_for_spikes[name]

        state.spike_midget_on_L = _bernoulli_spikes(_src("fr_midget_on_L"))
        state.spike_midget_off_L = _bernoulli_spikes(_src("fr_midget_off_L"))
        state.spike_midget_on_M = _bernoulli_spikes(_src("fr_midget_on_M"))
        state.spike_midget_off_M = _bernoulli_spikes(_src("fr_midget_off_M"))
        state.spike_parasol_on = _bernoulli_spikes(_src("fr_parasol_on"))
        state.spike_parasol_off = _bernoulli_spikes(_src("fr_parasol_off"))
    else:
        z = getattr(state, "spike_midget_on_L", None)
        if z is not None:
            for nm in (
                "spike_midget_on_L",
                "spike_midget_off_L",
                "spike_midget_on_M",
                "spike_midget_off_M",
                "spike_parasol_on",
                "spike_parasol_off",
            ):
                arr = getattr(state, nm, None)
                if arr is not None:
                    arr.fill(0.0)

    # 11. Mark all textures dirty
    for key in state.dirty_flags:
        state.dirty_flags[key] = True

