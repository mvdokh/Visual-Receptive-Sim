"""
Eccentricity-dependent scaling for large-field retina simulation.

Formulas from Watanabe & Rodieck (1989), Dacey (1993).
"""

from __future__ import annotations

import numpy as np

from src.simulation.bio_constants import MICRONS_PER_DEGREE


def get_eccentricity_deg(
    x_px: float,
    y_px: float,
    fovea_x: float,
    fovea_y: float,
    microns_per_px: float,
) -> float:
    """Distance from fovea center in degrees of visual angle."""
    dist_um = np.sqrt((x_px - fovea_x) ** 2 + (y_px - fovea_y) ** 2) * microns_per_px
    return dist_um / MICRONS_PER_DEGREE


def get_parasol_dend_radius_um(ecc_deg: float) -> float:
    """Parasol dendritic field radius in μm. Diameter ≈ 100 + 20 * ecc_deg (Watanabe & Rodieck 1989, Dacey 1993)."""
    diameter_um = 100.0 + 20.0 * ecc_deg
    return diameter_um / 2.0


def get_midget_dend_radius_um(ecc_deg: float) -> float:
    """Midget dendritic field radius in μm. Diameter ≈ 5 + 3 * ecc_deg."""
    diameter_um = 5.0 + 3.0 * ecc_deg
    return diameter_um / 2.0
