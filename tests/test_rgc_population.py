"""RGC population module: normalization, defaults, validation."""
from __future__ import annotations

import pytest

from src.config import default_config
from src.simulation.rgc_type_constants import RGC_TYPES, total_classified_fraction
from src.simulation.rgc_population import (
    SI_BASELINE_MIDGET,
    SI_BASELINE_PARASOL,
    calibrated_dendritic_sigmas_deg,
    compute_cross_type_rf_modulation,
    default_type_fractions,
    normalize_type_fractions,
    population_fractions_from_config,
    validate_population_against_paper,
)


def test_registry_coverage_sum():
    assert total_classified_fraction() == pytest.approx(0.89, rel=1e-3)
    assert len(RGC_TYPES) == 42


def test_normalize_fractions():
    base = default_type_fractions()
    assert sum(base.values()) == pytest.approx(1.0)
    noisy = dict(base)
    noisy["ON_alpha"] *= 3.0
    n = normalize_type_fractions(noisy)
    assert sum(n.values()) == pytest.approx(1.0)


def test_population_fractions_from_config_matches_defaults():
    cfg = default_config()
    tf = population_fractions_from_config(cfg.rgc_population)
    assert sum(tf.values()) == pytest.approx(1.0)
    ref = default_type_fractions()
    for k in RGC_TYPES:
        assert tf[k] == pytest.approx(ref[k], abs=1e-6)


def test_cross_type_modulation_identity_at_defaults():
    mod = compute_cross_type_rf_modulation(default_type_fractions())
    assert mod["gamma_aii_scale"] == pytest.approx(1.0, abs=0.05)
    assert mod["gamma_wide_scale"] == pytest.approx(1.0, abs=0.05)


def test_calibrated_sigmas_at_defaults():
    d = default_config()
    sm, sp, si_m, si_p, rsc, sur_m, sur_p = calibrated_dendritic_sigmas_deg(
        default_type_fractions(),
        d.dendritic.sigma_midget_deg,
        d.dendritic.sigma_parasol_deg,
        False,
    )
    assert sm == pytest.approx(d.dendritic.sigma_midget_deg, rel=1e-4)
    assert sp == pytest.approx(d.dendritic.sigma_parasol_deg, rel=1e-4)
    assert si_m == pytest.approx(0.0, abs=1e-6)
    assert si_p == pytest.approx(0.0, abs=1e-6)
    assert rsc == pytest.approx(1.0, abs=0.02)


def test_ds_validation_bounds():
    from src.simulation.rgc_type_constants import FUNCTIONAL_GROUPS

    tf = dict(default_type_fractions())
    for n in FUNCTIONAL_GROUPS["DS"]:
        tf[n] = 0.0
    tf = normalize_type_fractions(tf)
    w = validate_population_against_paper(tf)
    assert any("below 5%" in x for x in w)


def test_baseline_si_defined():
    assert 0.0 <= SI_BASELINE_MIDGET <= 1.0
    assert 0.0 <= SI_BASELINE_PARASOL <= 1.0
