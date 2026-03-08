from __future__ import annotations

"""
Bipolar cell helpers (midget and diffuse, ON and OFF).
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from src.config import GlobalConfig


def compute_bipolar_grids(
    cone_L_eff: np.ndarray,
    cone_M_eff: np.ndarray,
    cfg: GlobalConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute midget ON/OFF (L, M) and diffuse ON/OFF bipolar grids.
    """
    cone_lm_eff = cone_L_eff + cone_M_eff
    sigma_diffuse = cfg.bipolar.sigma_diffuse_deg / cfg.retina.dx_deg

    bp_midget_on_L = np.maximum(0.0, cone_L_eff)
    bp_midget_off_L = np.maximum(0.0, -cone_L_eff)
    bp_midget_on_M = np.maximum(0.0, cone_M_eff)
    bp_midget_off_M = np.maximum(0.0, -cone_M_eff)

    pooled = gaussian_filter(cone_lm_eff, sigma=sigma_diffuse, mode="reflect")
    bp_diffuse_on = np.maximum(0.0, pooled)
    bp_diffuse_off = np.maximum(0.0, -pooled)

    return (
        bp_midget_on_L,
        bp_midget_off_L,
        bp_midget_on_M,
        bp_midget_off_M,
        bp_diffuse_on,
        bp_diffuse_off,
    )

