from __future__ import annotations

"""
Amacrine cell helpers.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from rgc_simulator.config import GlobalConfig


def compute_amacrine(
    bp_midget_on_L: np.ndarray,
    bp_midget_on_M: np.ndarray,
    cone_lm_eff: np.ndarray,
    cfg: GlobalConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute AII and wide-field amacrine activity and total inhibition."""
    sigma_aii = cfg.amacrine.sigma_aii_deg / cfg.retina.dx_deg
    sigma_wide = cfg.amacrine.sigma_wide_deg / cfg.retina.dx_deg

    amacrine_aii = gaussian_filter(
        bp_midget_on_L + bp_midget_on_M, sigma=sigma_aii, mode="reflect"
    )
    amacrine_wide = gaussian_filter(cone_lm_eff, sigma=sigma_wide, mode="reflect")

    total = cfg.amacrine.gamma_aii * amacrine_aii + cfg.amacrine.gamma_wide * amacrine_wide
    return amacrine_aii, amacrine_wide, total

