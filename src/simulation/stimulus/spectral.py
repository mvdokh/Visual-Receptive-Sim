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
from skimage.transform import resize

from src.config import SpectralConfig, default_config
from src.simulation.rgb_mapping import build_emission_from_rgb, resample_spd_to_spectral_grid

# Monitor primaries from rgbtolms are in small radiometric units; Gaussian stimuli use peak ~1.
# Cache scales so RGB→SPD image stimuli match cone-drive magnitude of the Gaussian anchor (see below).
_RGBTOLMS_SCALE_CACHE: dict[tuple[int, int, int, str], float] = {}


def _rgbtolms_stimulus_scale(spectral: SpectralConfig) -> float:
    """
    Scale factor so image stimuli using rgbtolms have similar cone drive to other stimulus types.

    Gaussian-based stimuli (spot, full_field, legacy RGB basis) use profiles normalized to ~peak 1.
    ``build_emission_from_rgb`` SPDs are much smaller in absolute units, so without scaling the
    einsum with L/M/S fundamentals yields ~300× weaker drive and cones appear not to respond.
    We anchor white RGB (1,1,1) to the same max(L,M,S) linear response as a 550 nm Gaussian
    with unit peak (same profile as the fast full_field path).
    """
    key = (
        int(spectral.lambda_min),
        int(spectral.lambda_max),
        int(spectral.lambda_step),
        str(spectral.fundamentals_csv),
    )
    if key in _RGBTOLMS_SCALE_CACHE:
        return _RGBTOLMS_SCALE_CACHE[key]

    lam = spectral.wavelengths.astype(np.float32)
    SL = spectral.sens_L.astype(np.float32)
    SM = spectral.sens_M.astype(np.float32)
    SS = spectral.sens_S.astype(np.float32)

    profile = np.exp(-0.5 * ((lam - 550.0) / 10.0) ** 2).astype(np.float32)
    profile *= 1.0 / max(float(profile.max()), 1e-6)
    rL = float(np.dot(profile, SL))
    rM = float(np.dot(profile, SM))
    rS = float(np.dot(profile, SS))
    ref_mag = max(rL, rM, rS)

    wl_src, em = build_emission_from_rgb(np.ones((1, 1, 3), dtype=np.float32))
    spec_w = resample_spd_to_spectral_grid(wl_src, em, lam).reshape(-1)
    iL = float(np.dot(spec_w, SL))
    iM = float(np.dot(spec_w, SM))
    iS = float(np.dot(spec_w, SS))
    img_mag = max(iL, iM, iS)

    scale = ref_mag / max(img_mag, 1e-12)
    _RGBTOLMS_SCALE_CACHE[key] = scale
    return scale


def build_stimulus_spectrum(
    params: Dict[str, float] | None,
    spectral: SpectralConfig | None = None,
    grid_shape: Tuple[int, int] = (256, 256),
    time_s: float = 0.0,
    retina=None,
) -> np.ndarray:
    if spectral is None:
        spectral = default_config().spectral
    if params is None:
        params = {}
    if retina is None:
        retina = default_config().retina

    stim_type = params.get("type", "spot")
    mapping_mode = params.get("rgb_mapping_mode", getattr(spectral, "image_rgb_mapping", "rgbtolms"))
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

    # Use provided retina so stimulus scales with grid (1° = same across 256 or 2048)
    dx_deg = retina.dx_deg

    # Fast path: spot and full_field (no meshgrid) — Cython when available, else NumPy.
    # This always uses the legacy Gaussian spectral profile so that wavelength_nm
    # and intensity behave exactly as before, regardless of mapping mode.
    if stim_type in ("full_field", "spot"):
        lam = spectral.wavelengths.astype(np.float32)
        profile = np.exp(-0.5 * ((lam - wavelength_nm) / 10.0) ** 2).astype(np.float32)
        profile *= intensity / max(float(profile.max()), 1e-6)
        try:
            from hot_numerical.stimulus_fill import fill_spot_or_full
            fill_spot_or_full(
                h, w, dx_deg, x_deg, y_deg, radius_deg,
                profile, spectrum, full_field=(stim_type == "full_field"),
            )
            return spectrum
        except ImportError:
            pass
        # NumPy fast path (no meshgrid): R^2 via broadcasting
        ys = (np.arange(h, dtype=np.float32) - h / 2 + 0.5) * dx_deg - y_deg
        xs = (np.arange(w, dtype=np.float32) - w / 2 + 0.5) * dx_deg - x_deg
        r_sq = (ys[:, None] ** 2) + (xs[None, :] ** 2)
        if stim_type == "full_field":
            mask = np.ones((h, w), dtype=np.float32)
        else:
            mask = (r_sq <= radius_deg ** 2).astype(np.float32)
        spectrum[:] = mask[..., None] * profile[None, None, :]
        return spectrum

    xs = (np.arange(w) - w / 2 + 0.5) * dx_deg
    ys = (np.arange(h) - h / 2 + 0.5) * dx_deg
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
    elif stim_type == "expanding_ring":
        # Large-field: ring expanding from center (tests eccentricity-dependent RF)
        ring_speed_deg_s = float(params.get("ring_speed_deg_s", 2.0))
        ring_width_deg = float(params.get("ring_width_deg", 0.5))
        r_center = ring_speed_deg_s * time_s
        mask = ((R >= r_center - ring_width_deg / 2) & (R <= r_center + ring_width_deg / 2)).astype(np.float32)
    elif stim_type == "drifting_grating_full":
        # Full-field drifting grating (no mask; entire grid)
        freq = spatial_freq_cpd
        phase_rad = np.radians(phase_deg) + 2 * np.pi * (vx_deg_s or 0.0) * time_s
        mask = (0.5 + 0.5 * np.sin(2 * np.pi * freq * Xr + phase_rad)).astype(np.float32)
        mask = np.clip(mask, 0, 1)
    elif stim_type == "image":
        # Arbitrary image/photo stimulus with preserved RGB so that cone
        # fundamentals can bin the colors into L/M/S.
        image_mask = params.get("image_mask")
        if image_mask is None:
            return spectrum
        img = np.asarray(image_mask, dtype=np.float32)
        # Ensure shape (H, W, 3) in [0, 1].
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] >= 3:
            img = img[..., :3]
        else:
            return spectrum
        # Resize to current grid if needed.
        if img.shape[0] != h or img.shape[1] != w:
            img = resize(
                img,
                (h, w),
                order=1,
                mode="reflect",
                anti_aliasing=True,
                preserve_range=True,
            ).astype(np.float32)
        # Normalize from 0–255 if needed.
        vmax = float(np.max(img)) if img.size > 0 else 0.0
        if vmax > 1.0:
            img = img / 255.0
        img = np.clip(img, 0.0, 1.0)

        lam = spectral.wavelengths.astype(np.float32)
        if mapping_mode == "rgbtolms":
            wl_src, emission = build_emission_from_rgb(img)
            spectrum_img = resample_spd_to_spectral_grid(wl_src, emission, lam)
            spectrum[:] = spectrum_img * _rgbtolms_stimulus_scale(spectral)
            if intensity != 1.0:
                spectrum *= intensity
            return spectrum

        # Legacy Gaussian RGB basis mapping.
        basis_R = np.exp(-0.5 * ((lam - 610.0) / 15.0) ** 2)
        basis_G = np.exp(-0.5 * ((lam - 540.0) / 15.0) ** 2)
        basis_B = np.exp(-0.5 * ((lam - 450.0) / 15.0) ** 2)
        # Normalize each basis to max 1 so intensity slider is meaningful.
        for b in (basis_R, basis_G, basis_B):
            b_max = float(b.max())
            if b_max > 0:
                b /= b_max

        R_chan = img[..., 0]
        G_chan = img[..., 1]
        B_chan = img[..., 2]
        # Broadcast per-pixel RGB onto spectral basis.
        spectrum[:] = (
            R_chan[..., None] * basis_R[None, None, :]
            + G_chan[..., None] * basis_G[None, None, :]
            + B_chan[..., None] * basis_B[None, None, :]
        )
        # Apply global intensity slider.
        if intensity != 1.0:
            spectrum *= intensity
        return spectrum
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

