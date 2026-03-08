from __future__ import annotations

"""
Cone layer helpers.

For now, the main implementation of cone responses lives in
`rgc_simulator.simulation.pipeline`. This module provides small,
reusable functions that mirror that behavior and can be used for
experiments or future refactors.
"""

import numpy as np

from rgc_simulator.config import SpectralConfig


def compute_cone_responses(
    stimulus_spectrum: np.ndarray,
    spectral: SpectralConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute L, M, S cone responses from a stimulus spectrum grid.

    Parameters
    ----------
    stimulus_spectrum:
        Array of shape (H, W, L) where L is the number of wavelength bands.
    spectral:
        Spectral configuration with cone fundamentals.
    """
    stim = stimulus_spectrum
    sens_L = spectral.sens_L.astype(np.float32)
    sens_M = spectral.sens_M.astype(np.float32)
    sens_S = spectral.sens_S.astype(np.float32)

    cone_L = np.einsum("hwl,l->hw", stim, sens_L, optimize=True)
    cone_M = np.einsum("hwl,l->hw", stim, sens_M, optimize=True)
    cone_S = np.einsum("hwl,l->hw", stim, sens_S, optimize=True)
    return cone_L, cone_M, cone_S

