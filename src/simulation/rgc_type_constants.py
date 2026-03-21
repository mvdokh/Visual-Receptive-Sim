# =============================================================
# RGC type registry (mouse), aligned with Goetz et al. 2022 Cell Reports
# 42 morpho-transcriptomic types; classified coverage ~89% of RGC population.
# Population fractions are approximate (Eyewire museum + paper coverage estimates).
# =============================================================
from __future__ import annotations

from typing import Any, Dict, List

# Target sum of population_fraction across all types (classified fraction of total RGCs)
CLASSIFIED_COVERAGE_TARGET: float = 0.89

FUNCTIONAL_GROUPS: Dict[str, List[str]] = {
    "ON_sustained": ["ON_alpha", "PixON", "M2", "M1"],
    "OFF_sustained": [
        "OFF_alpha",
        "OFF_med_sus",
        "OFF_sus_EW1no",
        "OFF_sus_EW3o",
        "OFFhOS",
        "OFFvOS",
    ],
    "ON_transient": ["M6", "ON_tr_MeRF", "ON_tr_SmRF", "ON_tr_EW6t"],
    "OFF_transient": ["OFF_tr_alpha", "OFF_tr_MeRF", "OFF_tr_SmRF"],
    "ON_OS": ["ONhOS_SmRF", "ONvOS_SmRF", "ONhOS_LgRF", "ONvOS_LgRF"],
    "DS": [
        "ON_OFF_DS_dorsal",
        "ON_OFF_DS_ventral",
        "ON_OFF_DS_nasal",
        "ON_OFF_DS_temporal",
        "ON_DS_sus_1",
        "ON_DS_sus_2",
        "ON_DS_sus_3",
        "ON_DS_tr",
    ],
    "ON_OFF_small_RF": ["HD1", "HD2", "UHD", "LED", "F_mini_ON", "F_mini_OFF"],
    "SbC_other": [
        "bSbC",
        "sSbC_EW27",
        "sSbC_EW28",
        "ON_delayed",
        "ON_bursty",
        "Motion_sensor",
        "ON_small_OFF_large",
    ],
}

# Tusc5/Trarg1-positive T5-associated cluster (Figure 5B); high SI, transient, ChAT-band stratification
T5_RGC_TYPES: List[str] = [
    "OFF_tr_SmRF",
    "OFF_tr_MeRF",
    "HD1",
    "HD2",
    "UHD",
    "F_mini_ON",
    "ON_tr_SmRF",
    "ON_tr_MeRF",
    "ON_tr_EW6t",
]

# Paper-style distinct colors for 8 functional groups (UI + 3D connectivity)
FUNCTIONAL_GROUP_COLORS: Dict[str, tuple[float, float, float]] = {
    "ON_sustained": (0.90, 0.35, 0.25),
    "OFF_sustained": (0.35, 0.55, 0.95),
    "ON_transient": (0.95, 0.65, 0.20),
    "OFF_transient": (0.45, 0.30, 0.75),
    "ON_OS": (0.25, 0.85, 0.45),
    "DS": (0.95, 0.25, 0.55),
    "ON_OFF_small_RF": (0.55, 0.85, 0.90),
    "SbC_other": (0.65, 0.50, 0.35),
}

# Integer weights (anchors: ON/OFF alpha ~4%, ON_tr_MeRF ~8%, OFF_tr_alpha ~5%, HD1~5%, etc.)
# Sum scaled to CLASSIFIED_COVERAGE_TARGET.
_RAW_POPULATION_WEIGHTS: Dict[str, int] = {
    "ON_alpha": 40,
    "PixON": 10,
    "M2": 10,
    "M1": 10,
    "OFF_alpha": 40,
    "OFF_med_sus": 20,
    "OFF_sus_EW1no": 10,
    "OFF_sus_EW3o": 10,
    "OFFhOS": 15,
    "OFFvOS": 15,
    "M6": 10,
    "ON_tr_MeRF": 80,
    "ON_tr_SmRF": 30,
    "ON_tr_EW6t": 20,
    "OFF_tr_alpha": 50,
    "OFF_tr_MeRF": 30,
    "OFF_tr_SmRF": 20,
    "ONhOS_SmRF": 15,
    "ONvOS_SmRF": 15,
    "ONhOS_LgRF": 15,
    "ONvOS_LgRF": 15,
    "ON_OFF_DS_dorsal": 30,
    "ON_OFF_DS_ventral": 30,
    "ON_OFF_DS_nasal": 30,
    "ON_OFF_DS_temporal": 30,
    "ON_DS_sus_1": 20,
    "ON_DS_sus_2": 20,
    "ON_DS_sus_3": 20,
    "ON_DS_tr": 20,
    "HD1": 50,
    "HD2": 40,
    "UHD": 30,
    "LED": 30,
    "F_mini_ON": 25,
    "F_mini_OFF": 25,
    "bSbC": 10,
    "sSbC_EW27": 10,
    "sSbC_EW28": 10,
    "ON_delayed": 10,
    "ON_bursty": 10,
    "Motion_sensor": 10,
    "ON_small_OFF_large": 10,
}


def _rf_center_um(dendritic_field_um: float, override: float | None = None) -> float:
    if override is not None:
        return float(override)
    return float(dendritic_field_um) / 5.5


def _build_rgc_types() -> Dict[str, Dict[str, Any]]:
    total_w = float(sum(_RAW_POPULATION_WEIGHTS.values()))
    out: Dict[str, Dict[str, Any]] = {}
    for name, w in _RAW_POPULATION_WEIGHTS.items():
        frac = (w / total_w) * CLASSIFIED_COVERAGE_TARGET
        out[name] = {"population_fraction": frac}
    # Per-type metadata (defaults; literature-style placeholders)
    meta: Dict[str, Dict[str, Any]] = {
        "ON_alpha": {
            "functional_group": "ON_sustained",
            "response_polarity": "ON",
            "response_kinetics": "sustained",
            "dendritic_field_um": 225.0,
            "stratification": "inner_IPL",
            "surround_suppression_index": 0.3,
            "rf_center_sigma_um": 40.0,
            "rf_surround_sigma_um": 120.0,
            "peak_firing_hz": 120.0,
            "response_duration_s": 1.0,
            "tusc5_positive": False,
            "chat_band_stratification": False,
            "eyewire_type": "8w",
            "transcriptomic_cluster": "C43",
        },
        "OFF_alpha": {
            "functional_group": "OFF_sustained",
            "response_polarity": "OFF",
            "response_kinetics": "sustained",
            "dendritic_field_um": 220.0,
            "stratification": "outer_IPL",
            "surround_suppression_index": 0.32,
            "rf_center_sigma_um": 42.0,
            "rf_surround_sigma_um": 126.0,
            "peak_firing_hz": 115.0,
            "response_duration_s": 1.0,
            "tusc5_positive": False,
            "chat_band_stratification": False,
            "eyewire_type": "8w",
            "transcriptomic_cluster": "C40",
        },
    }
    # Template by functional group for bulk defaults
    group_defaults: Dict[str, Dict[str, Any]] = {
        "ON_sustained": {
            "response_polarity": "ON",
            "response_kinetics": "sustained",
            "dendritic_field_um": 80.0,
            "stratification": "inner_IPL",
            "surround_suppression_index": 0.25,
            "peak_firing_hz": 80.0,
            "response_duration_s": 1.2,
            "tusc5_positive": False,
            "chat_band_stratification": False,
            "eyewire_type": "mix",
            "transcriptomic_cluster": "C*",
        },
        "OFF_sustained": {
            "response_polarity": "OFF",
            "response_kinetics": "sustained",
            "dendritic_field_um": 85.0,
            "stratification": "outer_IPL",
            "surround_suppression_index": 0.28,
            "peak_firing_hz": 75.0,
            "response_duration_s": 1.1,
            "tusc5_positive": False,
            "chat_band_stratification": False,
            "eyewire_type": "mix",
            "transcriptomic_cluster": "C*",
        },
        "ON_transient": {
            "response_polarity": "ON",
            "response_kinetics": "transient",
            "dendritic_field_um": 140.0,
            "stratification": "middle_IPL",
            "surround_suppression_index": 0.45,
            "peak_firing_hz": 100.0,
            "response_duration_s": 0.15,
            "tusc5_positive": False,
            "chat_band_stratification": False,
            "eyewire_type": "mix",
            "transcriptomic_cluster": "C*",
        },
        "OFF_transient": {
            "response_polarity": "OFF",
            "response_kinetics": "transient",
            "dendritic_field_um": 135.0,
            "stratification": "middle_IPL",
            "surround_suppression_index": 0.48,
            "peak_firing_hz": 95.0,
            "response_duration_s": 0.15,
            "tusc5_positive": False,
            "chat_band_stratification": False,
            "eyewire_type": "mix",
            "transcriptomic_cluster": "C*",
        },
        "ON_OS": {
            "response_polarity": "ON",
            "response_kinetics": "sustained",
            "dendritic_field_um": 70.0,
            "stratification": "inner_IPL",
            "surround_suppression_index": 0.22,
            "peak_firing_hz": 70.0,
            "response_duration_s": 1.0,
            "tusc5_positive": False,
            "chat_band_stratification": False,
            "eyewire_type": "OS",
            "transcriptomic_cluster": "C*",
        },
        "DS": {
            "response_polarity": "ON_OFF",
            "response_kinetics": "transient",
            "dendritic_field_um": 100.0,
            "stratification": "bistratified",
            "surround_suppression_index": 0.5,
            "peak_firing_hz": 90.0,
            "response_duration_s": 0.2,
            "tusc5_positive": False,
            "chat_band_stratification": False,
            "eyewire_type": "DS",
            "transcriptomic_cluster": "C*",
        },
        "ON_OFF_small_RF": {
            "response_polarity": "ON_OFF",
            "response_kinetics": "transient",
            "dendritic_field_um": 55.0,
            "stratification": "bistratified",
            "surround_suppression_index": 0.55,
            "peak_firing_hz": 85.0,
            "response_duration_s": 0.12,
            "tusc5_positive": False,
            "chat_band_stratification": True,
            "eyewire_type": "HD",
            "transcriptomic_cluster": "C*",
        },
        "SbC_other": {
            "response_polarity": "ON",
            "response_kinetics": "sustained",
            "dendritic_field_um": 50.0,
            "stratification": "inner_IPL",
            "surround_suppression_index": 0.75,
            "peak_firing_hz": 40.0,
            "response_duration_s": 2.0,
            "tusc5_positive": False,
            "chat_band_stratification": False,
            "eyewire_type": "SbC",
            "transcriptomic_cluster": "C*",
        },
    }

    # Find functional group for each type name
    name_to_group: Dict[str, str] = {}
    for g, names in FUNCTIONAL_GROUPS.items():
        for n in names:
            name_to_group[n] = g

    for name in _RAW_POPULATION_WEIGHTS:
        g = name_to_group[name]
        base = dict(group_defaults[g])
        base["functional_group"] = g
        if name in meta:
            base.update(meta[name])
        # T5 members
        if name in T5_RGC_TYPES:
            base["tusc5_positive"] = True
            base["chat_band_stratification"] = True
            base["surround_suppression_index"] = max(
                float(base.get("surround_suppression_index", 0.5)), 0.52
            )
            base["response_kinetics"] = "transient"
        df = float(base["dendritic_field_um"])
        rc = base.get("rf_center_sigma_um")
        rs = base.get("rf_surround_sigma_um")
        if "rf_center_sigma_um" not in base:
            base["rf_center_sigma_um"] = _rf_center_um(df, None if rc is None else float(rc))
        if "rf_surround_sigma_um" not in base:
            c = float(base["rf_center_sigma_um"])
            base["rf_surround_sigma_um"] = 3.0 * c
        entry = {**base, "population_fraction": out[name]["population_fraction"]}
        out[name] = entry
    return out


RGC_TYPES: Dict[str, Dict[str, Any]] = _build_rgc_types()

ALL_RGC_TYPE_KEYS: List[str] = sorted(RGC_TYPES.keys())


def total_classified_fraction() -> float:
    return float(sum(t["population_fraction"] for t in RGC_TYPES.values()))
