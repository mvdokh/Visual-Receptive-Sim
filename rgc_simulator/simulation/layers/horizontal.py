from __future__ import annotations

"""
Horizontal cell helpers.

The main pipeline currently implements lateral pooling and feedback directly.
These helpers expose the same computations in isolation.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from rgc_simulator.config import GlobalConfig


def compute_horizontal_activation(
    cone_L: np.ndarray,
    cone_M: np.ndarray,
    cone_S: np.ndarray,
    cfg: GlobalConfig,
) -> np.ndarray:
    """Compute horizontal cell activation from cone signals."""
    cone_lm = cone_L + cone_M
    sigma_H = cfg.horizontal.sigma_lm_deg / cfg.retina.dx_deg
    sigma_H_s = cfg.horizontal.sigma_s_deg / cfg.retina.dx_deg

    h_lm = gaussian_filter(cone_lm, sigma=sigma_H, mode="reflect")
    h_s = gaussian_filter(cone_S, sigma=sigma_H_s, mode="reflect")
    return h_lm * cfg.horizontal.alpha_lm + h_s * cfg.horizontal.alpha_s


def apply_feedback(
    cone_L: np.ndarray,
    cone_M: np.ndarray,
    cone_S: np.ndarray,
    h_activation: np.ndarray,
    cfg: GlobalConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply horizontal feedback to cones, generating effective cone signals."""
    cone_L_eff = cone_L - cfg.horizontal.alpha_lm * h_activation
    cone_M_eff = cone_M - cfg.horizontal.alpha_lm * h_activation
    cone_S_eff = cone_S - cfg.horizontal.alpha_s * h_activation
    return cone_L_eff, cone_M_eff, cone_S_eff

