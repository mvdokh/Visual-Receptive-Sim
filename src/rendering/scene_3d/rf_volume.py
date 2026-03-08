from __future__ import annotations

"""
Receptive field volume representation for a selected cell.

The heavy-lift RF probe sweep and volume rendering are intentionally left
for later; this module defines the data shapes and parameters.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class RFVolume:
    grid: np.ndarray  # (H, W) RF map at stimulus plane
    center_sigma: float
    surround_sigma: float
    center_weight: float
    surround_weight: float
    peak_position: tuple[float, float]

