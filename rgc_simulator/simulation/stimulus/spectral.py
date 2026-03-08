from __future__ import annotations

"""
Spectral stimulus construction.

Stimulus types:
- spot: circular spot
- full_field: uniform field
- annulus: ring (hole in center)
- bar: oriented bar
- grating: sinusoidal grating
- checkerboard: alternating squares

Parameters:
- type: stimulus type
- wavelength_nm: center wavelength
- intensity: brightness (0–1)
- radius_deg: spot/annulus radius
- x_deg, y_deg: center position
- orientation_deg: bar/grating orientation (0=horizontal)
- width_deg: bar width
- spatial_freq_cpd: grating spatial frequency (cycles per degree)
- phase_deg: grating phase
- inner_radius_deg: annulus inner radius
"""

from typing import Dict, Tuple

import numpy as np

from rgc_simulator.config import SpectralConfig, default_config


def build_stimulus_spectrum(
    params: Dict[str, float] | None,
    spectral: SpectralConfig | None = None,
    grid_shape: Tuple[int, int] = (256, 256),
) -> np.ndarray:
    if spectral is None:
        spectral = default_config().spectral
    if params is None:
        params = {}

    stim_type = params.get("type", "spot")
    wavelength_nm = float(params.get("wavelength_nm", 550.0))
    intensity = float(params.get("intensity", 1.0))
    radius_deg = float(params.get("radius_deg", 0.15))
    x_deg = float(params.get("x_deg", 0.0))
    y_deg = float(params.get("y_deg", 0.0))
    orientation_deg = float(params.get("orientation_deg", 0.0))
    width_deg = float(params.get("width_deg", 0.1))
    spatial_freq_cpd = float(params.get("spatial_freq_cpd", 2.0))
    phase_deg = float(params.get("phase_deg", 0.0))
    inner_radius_deg = float(params.get("inner_radius_deg", 0.05))

    h, w = grid_shape
    spectrum = np.zeros((h, w, spectral.wavelengths.size), dtype=np.float32)

    cfg = default_config().retina
    xs = (np.arange(w) - w / 2 + 0.5) * cfg.dx_deg
    ys = (np.arange(h) - h / 2 + 0.5) * cfg.dx_deg
    X, Y = np.meshgrid(xs, ys)
    Xc = X - x_deg
    Yc = Y - y_deg
    R = np.sqrt(Xc**2 + Yc**2)

    # Rotation for oriented stimuli
    rad = np.radians(orientation_deg)
    Xr = Xc * np.cos(rad) + Yc * np.sin(rad)
    Yr = -Xc * np.sin(rad) + Yc * np.cos(rad)

    if stim_type == "full_field":
        mask = np.ones_like(R, dtype=np.float32)
    elif stim_type == "spot":
        mask = (R <= radius_deg).astype(np.float32)
    elif stim_type == "annulus":
        mask = ((R >= inner_radius_deg) & (R <= radius_deg)).astype(np.float32)
    elif stim_type == "ring":
        mask = ((R >= inner_radius_deg) & (R <= radius_deg)).astype(np.float32)
    elif stim_type == "bar":
        half_w = width_deg / 2
        mask = (np.abs(Yr) <= half_w).astype(np.float32)
    elif stim_type == "grating":
        # Xr is the axis along the grating
        freq = spatial_freq_cpd
        phase_rad = np.radians(phase_deg)
        mask = (0.5 + 0.5 * np.sin(2 * np.pi * freq * Xr + phase_rad)).astype(np.float32)
        mask = np.clip(mask, 0, 1)
    elif stim_type == "checkerboard":
        check_deg = width_deg
        ix = np.floor(Xc / check_deg).astype(int)
        iy = np.floor(Yc / check_deg).astype(int)
        mask = ((ix + iy) % 2 == 0).astype(np.float32)
    else:
        mask = (R <= radius_deg).astype(np.float32)

    lam = spectral.wavelengths
    spectral_profile = np.exp(-0.5 * ((lam - wavelength_nm) / 10.0) ** 2)
    spectral_profile *= intensity / max(float(spectral_profile.max()), 1e-6)

    spectrum[:] = mask[..., None] * spectral_profile[None, None, :]
    return spectrum

