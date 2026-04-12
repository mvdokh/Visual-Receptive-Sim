"""Spatial heterogeneity: inactive parameters must not affect homogeneous output."""
from __future__ import annotations

import numpy as np

from src.config import SpatialHeterogeneityMode, default_config
from src.simulation.pipeline import tick
from src.simulation.state import SimState


def test_homogeneous_invariant_to_stored_scatter_and_mosaic_params():
    """Mode HOMOGENEOUS ignores scatter sigma and mosaic N; firing maps match baseline."""
    s1 = SimState(config=default_config())
    s2 = SimState(config=default_config())
    sh2 = s2.config.spatial_heterogeneity
    sh2.scatter.sigma = 0.99
    sh2.scatter.affect_cone_to_bipolar = True
    sh2.scatter.affect_bipolar_to_rgc = True
    sh2.scatter.affect_amacrine_to_bipolar = True
    sh2.mosaic.n_cells = 1800
    sh2.type_map.type_fractions = (0.5, 0.1, 0.1, 0.1, 0.1, 0.1)
    for s in (s1, s2):
        s.ensure_initialized()
        s.stimulus_params = {"type": "spot", "intensity": 1.0}
        assert s.config.spatial_heterogeneity.mode == SpatialHeterogeneityMode.HOMOGENEOUS
        tick(s, 0.05)
    np.testing.assert_array_almost_equal(s1.fr_midget_on_L, s2.fr_midget_on_L, decimal=5)
