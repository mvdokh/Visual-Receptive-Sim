# =============================================================
# RGC population composition → effective RF and circuit modulation.
# Phenomenological coupling to the existing midget/parasol pipeline.
# =============================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from src.simulation.bio_constants import MICRONS_PER_DEGREE
from src.simulation.rgc_type_constants import (
    FUNCTIONAL_GROUP_COLORS,
    FUNCTIONAL_GROUPS,
    RGC_TYPES,
    T5_RGC_TYPES,
)


# Map functional groups to (midget_weight, parasol_weight) for RF pooling (sums arbitrary; normalized per use)
GROUP_PATHWAY_WEIGHTS: Dict[str, Tuple[float, float]] = {
    "ON_sustained": (0.55, 0.45),
    "OFF_sustained": (0.50, 0.50),
    "ON_transient": (0.35, 0.65),
    "OFF_transient": (0.35, 0.65),
    "ON_OS": (0.75, 0.25),
    "DS": (0.40, 0.60),
    "ON_OFF_small_RF": (0.65, 0.35),
    "SbC_other": (0.80, 0.20),
}

# Baseline LN firing (Hz); ON alpha reference for r_max scaling
PEAK_FIRING_REFERENCE_HZ: float = 120.0


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# (source_label, target_key_or_group, direction, weight_range_low_high)
# target can be a type name or a functional group name prefixed with "group:"
CIRCUIT_INTERACTIONS: List[Tuple[str, str, int, Tuple[float, float]]] = [
    ("AII_amacrine", "OFF_tr_alpha", -1, (0.3, 0.8)),
    ("wide_field_amacrine", "ON_alpha", -1, (0.1, 0.5)),
    ("horizontal_cell", "group:OFF_sustained", -1, (0.2, 0.6)),
    ("OFF_OS", "F_mini_ON", 1, (0.1, 0.3)),
]


def _mid(r: Tuple[float, float]) -> float:
    return 0.5 * (r[0] + r[1])


def default_type_fractions() -> Dict[str, float]:
    """Normalize registry population_fraction to sum to 1.0 over all types."""
    raw = {k: float(v["population_fraction"]) for k, v in RGC_TYPES.items()}
    s = sum(raw.values()) or 1.0
    return {k: raw[k] / s for k in raw}


DEFAULT_TYPE_FRACTIONS: Dict[str, float] = default_type_fractions()


def population_fractions_from_config(rpc: Any) -> Dict[str, float]:
    """Combine registry defaults, group scales, per-type multipliers, and regional toggles."""
    base = default_type_fractions()
    raw: Dict[str, float] = {}
    for n in RGC_TYPES:
        g = str(RGC_TYPES[n]["functional_group"])
        raw[n] = (
            base[n]
            * float(rpc.group_scales.get(g, 1.0))
            * float(rpc.type_weight_multipliers.get(n, 1.0))
        )
    if getattr(rpc, "dorsal_retina_mode", False):
        raw["PixON"] = raw.get("PixON", 0.0) * 1.45
    if getattr(rpc, "ventral_retina_mode", False):
        raw["F_mini_ON"] = raw.get("F_mini_ON", 0.0) * 1.45
    return normalize_type_fractions(raw)


def group_population_shares(type_fractions: Mapping[str, float]) -> Dict[str, float]:
    """Sum of normalized type fractions per functional group."""
    tf = normalize_type_fractions(type_fractions)
    return {g: sum(tf.get(n, 0.0) for n in names) for g, names in FUNCTIONAL_GROUPS.items()}


def normalize_type_fractions(fracs: Mapping[str, float]) -> Dict[str, float]:
    keys = list(RGC_TYPES.keys())
    out = {k: float(fracs.get(k, 0.0)) for k in keys}
    s = sum(out.values())
    if s <= 0:
        return dict(DEFAULT_TYPE_FRACTIONS)
    return {k: out[k] / s for k in keys}


@dataclass
class RGCPopulationState:
    type_fractions: Dict[str, float]
    total_rgc_density: float = 8000.0  # cells/mm²
    region: str = "parafovea"  # fovea | parafovea | periphery


@dataclass
class GroupEffectiveRF:
    """Population-weighted effective RF metrics per functional group."""

    functional_group: str
    center_sigma_um: float
    surround_sigma_um: float
    surround_suppression_index: float
    dominant_kinetics: str
    peak_firing_hz: float
    needs_on_off_components: bool
    population_share: float  # share of total modelled population in this group


def um_to_deg(sigma_um: float) -> float:
    return float(sigma_um) / float(MICRONS_PER_DEGREE)


def deg_to_um(sigma_deg: float) -> float:
    return float(sigma_deg) * float(MICRONS_PER_DEGREE)


def _type_effective_row(
    name: str,
    type_fractions: Mapping[str, float],
    t5_bias: bool,
) -> Dict[str, Any]:
    t = RGC_TYPES[name]
    ft = float(type_fractions.get(name, 0.0))
    si = float(t["surround_suppression_index"])
    if t5_bias and name in T5_RGC_TYPES:
        si = max(si, 0.51)
    kin = str(t.get("response_kinetics", "sustained"))
    if t5_bias and name in T5_RGC_TYPES:
        kin = "transient"
    return {
        "ft": ft,
        "center_um": float(t["rf_center_sigma_um"]),
        "surround_um": float(t["rf_surround_sigma_um"]),
        "si": si,
        "peak_hz": float(t["peak_firing_hz"]),
        "kinetics": kin,
        "strat": str(t.get("stratification", "inner_IPL")),
        "polarity": str(t.get("response_polarity", "ON")),
    }


def compute_effective_rf(
    population_state: RGCPopulationState,
    t5_cluster_bias: bool = False,
) -> Dict[str, GroupEffectiveRF]:
    """
    Per-functional-group weighted RF metrics.
    Types are weighted by their share within the group, then groups weighted by population share.
    """
    tf = normalize_type_fractions(population_state.type_fractions)
    group_shares: Dict[str, float] = {}
    for g, names in FUNCTIONAL_GROUPS.items():
        group_shares[g] = sum(tf.get(n, 0.0) for n in names)

    out: Dict[str, GroupEffectiveRF] = {}
    for g, names in FUNCTIONAL_GROUPS.items():
        wg = group_shares[g]
        if wg <= 1e-12:
            # fall back to uniform within group from registry
            inner = {n: 1.0 / len(names) for n in names}
        else:
            inner = {n: tf.get(n, 0.0) / wg for n in names}

        centers: List[float] = []
        surrounds: List[float] = []
        sis: List[float] = []
        peaks: List[float] = []
        kin_votes: Dict[str, float] = {}
        for n in names:
            row = _type_effective_row(n, tf, t5_cluster_bias)
            w = inner[n]
            centers.append(row["center_um"] * w)
            surrounds.append(row["surround_um"] * w)
            sis.append(row["si"] * w)
            peaks.append(row["peak_hz"] * w)
            ek = str(row["kinetics"])
            kin_votes[ek] = kin_votes.get(ek, 0.0) + w

        c_um = sum(centers)
        s_um = sum(surrounds)
        si_eff = sum(sis)
        pk = sum(peaks)
        dominant_kinetics = (
            max(kin_votes.items(), key=lambda x: x[1])[0] if kin_votes else "sustained"
        )

        if g == "SbC_other":
            si_eff = max(si_eff, 0.65)

        needs_on = g in ("DS", "ON_OFF_small_RF")
        out[g] = GroupEffectiveRF(
            functional_group=g,
            center_sigma_um=c_um,
            surround_sigma_um=s_um,
            surround_suppression_index=si_eff,
            dominant_kinetics=dominant_kinetics,
            peak_firing_hz=pk,
            needs_on_off_components=needs_on,
            population_share=wg,
        )
    return out


def _fraction_for_target(
    target: str,
    type_fractions: Mapping[str, float],
) -> float:
    if target.startswith("group:"):
        g = target.split(":", 1)[1]
        names = FUNCTIONAL_GROUPS.get(g, [])
        return sum(float(type_fractions.get(n, 0.0)) for n in names)
    return float(type_fractions.get(target, 0.0))


def compute_cross_type_rf_modulation(
    type_fractions: Mapping[str, float],
    reference_fractions: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """
    Unitless multipliers for amacrine horizontal coupling vs baseline reference fractions.
    Returns keys: gamma_aii_scale, gamma_wide_scale, horizontal_alpha_lm_scale (optional use).
    """
    ref = reference_fractions if reference_fractions is not None else DEFAULT_TYPE_FRACTIONS
    tf = normalize_type_fractions(type_fractions)
    gamma_aii = 1.0
    gamma_wide = 1.0
    h_alpha = 1.0

    for src, tgt, direction, wrange in CIRCUIT_INTERACTIONS:
        mid = _mid(wrange)
        f_t = _fraction_for_target(tgt, tf)
        f0 = _fraction_for_target(tgt, ref)
        delta = f_t - f0
        contrib = 1.0 + direction * mid * delta * 2.0

        if src == "AII_amacrine":
            gamma_aii *= _clip(contrib, 0.75, 1.25)
        elif src == "wide_field_amacrine":
            gamma_wide *= _clip(contrib, 0.85, 1.15)
        elif src == "horizontal_cell":
            h_alpha *= _clip(contrib, 0.8, 1.2)
        elif src == "OFF_OS":
            gamma_wide *= _clip(contrib, 0.9, 1.1)

    return {
        "gamma_aii_scale": gamma_aii,
        "gamma_wide_scale": gamma_wide,
        "horizontal_alpha_lm_scale": h_alpha,
    }


def pathway_um_and_si(
    type_fractions: Mapping[str, float],
    effective_by_group: Mapping[str, GroupEffectiveRF],
) -> Tuple[float, float, float, float]:
    """
    Returns (midget_center_um, parasol_center_um, si_midget, si_parasol).
    Weighted by group population share and GROUP_PATHWAY_WEIGHTS.
    """
    tf = normalize_type_fractions(type_fractions)
    group_shares: Dict[str, float] = {
        g: sum(tf.get(n, 0.0) for n in names) for g, names in FUNCTIONAL_GROUPS.items()
    }

    m_num = m_den = p_num = p_den = 0.0
    si_m_num = si_m_den = si_p_num = si_p_den = 0.0
    for g, eff in effective_by_group.items():
        wg = group_shares.get(g, 0.0)
        if wg <= 0:
            continue
        wm, wp = GROUP_PATHWAY_WEIGHTS.get(g, (0.5, 0.5))
        m_num += wg * wm * eff.center_sigma_um
        m_den += wg * wm
        p_num += wg * wp * eff.center_sigma_um
        p_den += wg * wp
        si_m_num += wg * wm * eff.surround_suppression_index
        si_m_den += wg * wm
        si_p_num += wg * wp * eff.surround_suppression_index
        si_p_den += wg * wp

    m_um = m_num / max(m_den, 1e-9)
    p_um = p_num / max(p_den, 1e-9)
    si_m = si_m_num / max(si_m_den, 1e-9)
    si_p = si_p_num / max(si_p_den, 1e-9)
    return (m_um, p_um, si_m, si_p)


def weighted_peak_firing_hz(type_fractions: Mapping[str, float]) -> float:
    tf = normalize_type_fractions(type_fractions)
    return sum(
        tf.get(n, 0.0) * float(RGC_TYPES[n]["peak_firing_hz"]) for n in RGC_TYPES
    )


def _baseline_pathway_um() -> Tuple[float, float, float, float]:
    """Reference center sigmas and SI at default registry fractions for calibration."""
    eff = compute_effective_rf(RGCPopulationState(type_fractions=DEFAULT_TYPE_FRACTIONS), False)
    m, p, si_m, si_p = pathway_um_and_si(DEFAULT_TYPE_FRACTIONS, eff)
    return (m, p, si_m, si_p)


BASELINE_MIDGET_CENTER_UM, BASELINE_PARASOL_CENTER_UM, SI_BASELINE_MIDGET, SI_BASELINE_PARASOL = (
    _baseline_pathway_um()
)

# Population-weighted peak rate at registry defaults (for r_max calibration)
BASELINE_PEAK_HZ: float = weighted_peak_firing_hz(DEFAULT_TYPE_FRACTIONS)


def calibrated_dendritic_sigmas_deg(
    type_fractions: Mapping[str, float],
    default_midget_deg: float,
    default_parasol_deg: float,
    t5_cluster_bias: bool,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Returns (sigma_midget_deg, sigma_parasol_deg, si_midget_rel, si_parasol_rel,
             r_max_scale, sigma_surround_midget_deg, sigma_surround_parasol_deg).
    si_*_rel are nonnegative surround weights relative to baseline (0 at default fractions).
    Surround sigmas = 3 × center sigmas in µm, converted to degrees.
    """
    st = RGCPopulationState(type_fractions=dict(type_fractions))
    eff = compute_effective_rf(st, t5_cluster_bias)
    m_um, p_um, si_m, si_p = pathway_um_and_si(type_fractions, eff)
    m_scale = m_um / max(BASELINE_MIDGET_CENTER_UM, 1e-9)
    p_scale = p_um / max(BASELINE_PARASOL_CENTER_UM, 1e-9)
    peak = weighted_peak_firing_hz(type_fractions)
    r_scale = peak / max(BASELINE_PEAK_HZ, 1e-9)
    si_m_rel = max(0.0, si_m - SI_BASELINE_MIDGET)
    si_p_rel = max(0.0, si_p - SI_BASELINE_PARASOL)
    # Surround σ = 3 × center σ (µm), same scale factors as center
    sur_m_um = 3.0 * m_um
    sur_p_um = 3.0 * p_um
    sur_m_deg = um_to_deg(sur_m_um)
    sur_p_deg = um_to_deg(sur_p_um)
    return (
        default_midget_deg * m_scale,
        default_parasol_deg * p_scale,
        si_m_rel,
        si_p_rel,
        r_scale,
        sur_m_deg,
        sur_p_deg,
    )


def validate_population_against_paper(
    type_fractions: Mapping[str, float],
    *,
    dorsal_mode: bool = False,
    ventral_mode: bool = False,
    t5_cluster_bias: bool = False,
) -> List[str]:
    """Return human-readable warnings (empty if all checks pass)."""
    warnings: List[str] = []
    tf = normalize_type_fractions(type_fractions)

    ds_names = FUNCTIONAL_GROUPS["DS"]
    f_ds = sum(tf.get(n, 0.0) for n in ds_names)
    if f_ds < 0.05:
        warnings.append(
            f"DS functional group is {f_ds:.1%} of population; below 5% (unusual)."
        )
    if f_ds > 0.40:
        warnings.append(
            f"DS functional group is {f_ds:.1%} of population; above 40% (unusual)."
        )

    if not dorsal_mode and tf.get("PixON", 0.0) > 0.05:
        warnings.append("PixON fraction is high; enable dorsal retina mode if intentional.")
    if not ventral_mode and tf.get("F_mini_ON", 0.0) > 0.05:
        warnings.append("F-mini-ON fraction is high; enable ventral retina mode if intentional.")

    if t5_cluster_bias:
        for n in T5_RGC_TYPES:
            k = str(RGC_TYPES[n].get("response_kinetics", ""))
            if k == "sustained":
                warnings.append(f"T5-biased type {n} is marked sustained (expect transient).")
    return warnings


def dominant_functional_group_for_pathway(
    pathway: str,
    type_fractions: Mapping[str, float],
) -> str:
    """
    pathway: 'midget' or 'parasol' — group with largest contribution to that pathway.
    """
    tf = normalize_type_fractions(type_fractions)
    best_g = "ON_sustained"
    best_score = -1.0
    for g, names in FUNCTIONAL_GROUPS.items():
        wg = sum(tf.get(n, 0.0) for n in names)
        wm, wp = GROUP_PATHWAY_WEIGHTS.get(g, (0.5, 0.5))
        score = wg * (wm if pathway == "midget" else wp)
        if score > best_score:
            best_score = score
            best_g = g
    return best_g


def functional_group_color_rgb(group: str) -> Tuple[float, float, float]:
    return FUNCTIONAL_GROUP_COLORS.get(group, (1.0, 0.27, 0.27))


def bipolar_to_rgc_line_color(
    type_fractions: Mapping[str, float],
    *,
    use_parasol_pathway: bool = True,
) -> Tuple[float, float, float]:
    """RGB for 3D bipolar→RGC lines; prefers parasol pathway to match diffuse bipolar input."""
    g = dominant_functional_group_for_pathway(
        "parasol" if use_parasol_pathway else "midget", type_fractions
    )
    return functional_group_color_rgb(g)


def line_color_for_rgc_population(rpc: Any) -> Tuple[float, float, float]:
    """Use full population_fractions_from_config (group scales × type weights × regional)."""
    tf = population_fractions_from_config(rpc)
    return bipolar_to_rgc_line_color(tf, use_parasol_pathway=True)
