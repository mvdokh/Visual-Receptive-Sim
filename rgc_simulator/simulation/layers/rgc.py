from __future__ import annotations

"""
Retinal ganglion cell helpers.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from rgc_simulator.config import GlobalConfig


def rgc_generator(
    bp_grid: np.ndarray,
    total_amacrine: np.ndarray,
    sigma_deg: float,
    cfg: GlobalConfig,
) -> np.ndarray:
    """Spatial integration via Gaussian dendritic field."""
    sigma_px = sigma_deg / cfg.retina.dx_deg
    return gaussian_filter(bp_grid - total_amacrine, sigma=sigma_px, mode="reflect")


def sigmoid(x: np.ndarray, r_max: float, x_half: float, slope: float) -> np.ndarray:
    """LN firing-rate nonlinearity."""
    return r_max / (1.0 + np.exp(-slope * (x - x_half)))

