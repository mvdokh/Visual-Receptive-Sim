from __future__ import annotations

import numpy as np

from src.config import default_config
from src.simulation.rgb_mapping import build_emission_from_rgb, resample_spd_to_spectral_grid
from src.simulation.stimulus.spectral import build_stimulus_spectrum


def test_build_emission_from_rgb_shape() -> None:
    rgb = np.array([[[0.2, 0.4, 0.6]]], dtype=np.float32)
    wl, spd = build_emission_from_rgb(rgb)
    assert wl.ndim == 1
    assert spd.shape == (1, 1, wl.shape[0])
    assert np.all(np.isfinite(spd))


def test_stimulus_spectrum_mapping_modes_image() -> None:
    cfg = default_config()
    h, w = 8, 8
    img = np.linspace(0.0, 1.0, h * w * 3, dtype=np.float32).reshape(h, w, 3)

    params_legacy = {
        "type": "image",
        "image_mask": img,
        "intensity": 1.0,
        "rgb_mapping_mode": "legacy_spectral",
    }
    spec_legacy = build_stimulus_spectrum(params_legacy, cfg.spectral, (h, w), 0.0, cfg.retina)

    params_rgbtolms = dict(params_legacy)
    params_rgbtolms["rgb_mapping_mode"] = "rgbtolms"
    spec_rgbtolms = build_stimulus_spectrum(params_rgbtolms, cfg.spectral, (h, w), 0.0, cfg.retina)

    assert spec_legacy.shape == spec_rgbtolms.shape == (h, w, cfg.spectral.wavelengths.size)
    assert np.isfinite(spec_legacy).all()
    assert np.isfinite(spec_rgbtolms).all()
    # The two mappings should not be identical for a generic RGB pattern.
    assert not np.allclose(spec_legacy, spec_rgbtolms)

