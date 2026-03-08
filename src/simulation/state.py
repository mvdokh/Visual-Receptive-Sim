from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from src.config import GlobalConfig, default_config


LayerGrid = np.ndarray  # alias: (H, W) float32


@dataclass
class SimState:
    """
    Shared simulation state.

    All layer activations are 2D grids in retinal coordinates.
    """

    config: GlobalConfig = field(default_factory=default_config)

    time: float = 0.0

    # Stimulus
    stimulus_params: Dict[str, float] = field(default_factory=dict)
    stimulus_spectrum: np.ndarray | None = None  # (H, W, L)

    # Cones (effective signals after H feedback)
    cone_L: LayerGrid | None = None
    cone_M: LayerGrid | None = None
    cone_S: LayerGrid | None = None
    cone_L_eff: LayerGrid | None = None
    cone_M_eff: LayerGrid | None = None
    cone_S_eff: LayerGrid | None = None

    # Horizontal
    h_activation: LayerGrid | None = None

    # Bipolar
    bp_midget_on_L: LayerGrid | None = None
    bp_midget_off_L: LayerGrid | None = None
    bp_midget_on_M: LayerGrid | None = None
    bp_midget_off_M: LayerGrid | None = None
    bp_diffuse_on: LayerGrid | None = None
    bp_diffuse_off: LayerGrid | None = None

    # Amacrine
    amacrine_aii: LayerGrid | None = None
    amacrine_wide: LayerGrid | None = None

    # RGC generators
    rgc_midget_on_L: LayerGrid | None = None
    rgc_midget_off_L: LayerGrid | None = None
    rgc_midget_on_M: LayerGrid | None = None
    rgc_midget_off_M: LayerGrid | None = None
    rgc_parasol_on: LayerGrid | None = None
    rgc_parasol_off: LayerGrid | None = None

    # Firing rates
    fr_midget_on_L: LayerGrid | None = None
    fr_midget_off_L: LayerGrid | None = None
    fr_midget_on_M: LayerGrid | None = None
    fr_midget_off_M: LayerGrid | None = None
    fr_parasol_on: LayerGrid | None = None
    fr_parasol_off: LayerGrid | None = None

    # Color opponent signals
    lm_opponent: LayerGrid | None = None
    by_opponent: LayerGrid | None = None

    # Smoothed versions for temporal dynamics
    smoothed: Dict[str, LayerGrid] = field(default_factory=dict)

    # Dirty flags to notify renderer which textures need updates
    dirty_flags: Dict[str, bool] = field(default_factory=dict)

    def grid_shape(self) -> Tuple[int, int]:
        n = self.config.retina.grid_resolution
        return n, n

    def ensure_initialized(self) -> None:
        """Allocate empty arrays if they have not been created yet."""
        h, w = self.grid_shape()
        zero = lambda: np.zeros((h, w), dtype=np.float32)

        for name in [
            "cone_L",
            "cone_M",
            "cone_S",
            "cone_L_eff",
            "cone_M_eff",
            "cone_S_eff",
            "h_activation",
            "bp_midget_on_L",
            "bp_midget_off_L",
            "bp_midget_on_M",
            "bp_midget_off_M",
            "bp_diffuse_on",
            "bp_diffuse_off",
            "amacrine_aii",
            "amacrine_wide",
            "rgc_midget_on_L",
            "rgc_midget_off_L",
            "rgc_midget_on_M",
            "rgc_midget_off_M",
            "rgc_parasol_on",
            "rgc_parasol_off",
            "fr_midget_on_L",
            "fr_midget_off_L",
            "fr_midget_on_M",
            "fr_midget_off_M",
            "fr_parasol_on",
            "fr_parasol_off",
            "lm_opponent",
            "by_opponent",
        ]:
            if getattr(self, name) is None:
                setattr(self, name, zero())

        # Initialize smoothed copies and dirty flags.
        for name in [
            "cone_L",
            "cone_M",
            "cone_S",
            "h_activation",
            "bp_midget_on_L",
            "bp_midget_off_L",
            "bp_midget_on_M",
            "bp_midget_off_M",
            "bp_diffuse_on",
            "bp_diffuse_off",
            "amacrine_aii",
            "amacrine_wide",
            "rgc_midget_on_L",
            "rgc_midget_off_L",
            "rgc_midget_on_M",
            "rgc_midget_off_M",
            "rgc_parasol_on",
            "rgc_parasol_off",
            "fr_midget_on_L",
            "fr_midget_off_L",
            "fr_midget_on_M",
            "fr_midget_off_M",
            "fr_parasol_on",
            "fr_parasol_off",
        ]:
            if name not in self.smoothed:
                self.smoothed[name] = zero()
            if name not in self.dirty_flags:
                self.dirty_flags[name] = True

