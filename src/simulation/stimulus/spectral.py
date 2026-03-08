from __future__ import annotations

"""
Spectral stimulus construction.

Stimulus types (spatial pattern + spectrum):
- spot: circular spot
- full_field: uniform field
- annulus: ring (hole in center)
- bar: oriented bar
- grating: sinusoidal grating
- checkerboard: alternating squares
- moving_spot: spot translating over time
- moving_bar: bar translating over time
- moving_grating: drifting grating
- dual_spot: two independent, possibly different-color spots

Parameters:
- type: stimulus type
- wavelength_nm: center wavelength (primary object)
- intensity: brightness (0–1) for primary object
- radius_deg: spot/annulus radius
- x_deg, y_deg: center position (at t = 0 for moving_* types)
- orientation_deg: bar/grating orientation (0=horizontal)
- width_deg: bar width or checker size
- spatial_freq_cpd: grating spatial frequency (cycles per degree)
- phase_deg: grating phase
- inner_radius_deg: annulus inner radius
- vx_deg_s, vy_deg_s: velocity (deg/s) for moving_* types
- wavelength2_nm, intensity2: secondary color for dual_spot
- x2_deg, y2_deg, radius2_deg: secondary spot geometry for dual_spot
"""

from typing import Dict, Tuple

import numpy as np

from src.config import SpectralConfig, default_config


def build_stimulus_spectrum(
    params: Dict[str, float] | None,
    spectral: SpectralConfig | None = None,
    grid_shape: Tuple[int, int] = (256, 256),
    time_s: float = 0.0,
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
    vx_deg_s = float(params.get("vx_deg_s", 0.0))
    vy_deg_s = float(params.get("vy_deg_s", 0.0))

    # Secondary object (dual_spot)
    wavelength2_nm = float(params.get("wavelength2_nm", 450.0))
    intensity2 = float(params.get("intensity2", 1.0))
    radius2_deg = float(params.get("radius2_deg", radius_deg))
    x2_deg = float(params.get("x2_deg", x_deg + radius_deg * 1.5))
    y2_deg = float(params.get("y2_deg", y_deg))

    h, w = grid_shape
    spectrum = np.zeros((h, w, spectral.wavelengths.size), dtype=np.float32)

    cfg = default_config().retina
    xs = (np.arange(w) - w / 2 + 0.5) * cfg.dx_deg
    ys = (np.arange(h) - h / 2 + 0.5) * cfg.dx_deg
    X, Y = np.meshgrid(xs, ys)

    # Time-dependent center for moving stimuli
    cx = x_deg
    cy = y_deg
    if stim_type in {"moving_spot", "moving_bar", "moving_grating"}:
        cx = x_deg + vx_deg_s * time_s
        cy = y_deg + vy_deg_s * time_s

    Xc = X - cx
    Yc = Y - cy
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
    elif stim_type == "grating" or stim_type == "moving_grating":
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
    elif stim_type == "moving_spot":
        mask = (R <= radius_deg).astype(np.float32)
    elif stim_type == "moving_bar":
        half_w = width_deg / 2
        mask = (np.abs(Yr) <= half_w).astype(np.float32)
    elif stim_type == "dual_spot":
        # Two independent, potentially different-color spots
        # Primary spot
        Xc1 = X - x_deg
        Yc1 = Y - y_deg
        R1 = np.sqrt(Xc1**2 + Yc1**2)
        mask1 = (R1 <= radius_deg).astype(np.float32)
        # Secondary spot
        Xc2 = X - x2_deg
        Yc2 = Y - y2_deg
        R2 = np.sqrt(Xc2**2 + Yc2**2)
        mask2 = (R2 <= radius2_deg).astype(np.float32)

        lam = spectral.wavelengths
        profile1 = np.exp(-0.5 * ((lam - wavelength_nm) / 10.0) ** 2)
        profile1 *= intensity / max(float(profile1.max()), 1e-6)
        profile2 = np.exp(-0.5 * ((lam - wavelength2_nm) / 10.0) ** 2)
        profile2 *= intensity2 / max(float(profile2.max()), 1e-6)

        spectrum[:] = (
            mask1[..., None] * profile1[None, None, :]
            + mask2[..., None] * profile2[None, None, :]
        )
        return spectrum
    else:
        mask = (R <= radius_deg).astype(np.float32)

    lam = spectral.wavelengths
    spectral_profile = np.exp(-0.5 * ((lam - wavelength_nm) / 10.0) ** 2)
    spectral_profile *= intensity / max(float(spectral_profile.max()), 1e-6)

    spectrum[:] = mask[..., None] * spectral_profile[None, None, :]
    return spectrum

