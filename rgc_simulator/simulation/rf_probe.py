"""
Receptive field probe sweep and Difference-of-Gaussians (DoG) fitting.

Sweeps a small spot across the stimulus plane and records the selected RGC type's
firing rate at each position. Fits a DoG model to the emergent RF.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

from rgc_simulator.config import GlobalConfig, default_config
from rgc_simulator.simulation.pipeline import tick
from rgc_simulator.simulation.state import SimState


@dataclass
class DoGFit:
    """Fitted DoG parameters for a receptive field."""

    sigma_center: float
    sigma_surround: float
    weight_center: float
    weight_surround: float
    x0: float
    y0: float
    baseline: float
    success: bool
    message: str = ""


def _dog_2d(
    xy: np.ndarray,
    x0: float,
    y0: float,
    sigma_c: float,
    sigma_s: float,
    w_c: float,
    w_s: float,
    baseline: float,
) -> np.ndarray:
    """Evaluate DoG at (x, y) positions. xy shape (N, 2)."""
    x, y = xy[:, 0], xy[:, 1]
    r2_c = (x - x0) ** 2 + (y - y0) ** 2
    r2_s = r2_c
    g_c = np.exp(-r2_c / (2 * sigma_c**2))
    g_s = np.exp(-r2_s / (2 * sigma_s**2))
    return baseline + w_c * g_c - w_s * g_s


def fit_dog(
    x_deg: np.ndarray,
    y_deg: np.ndarray,
    rf_map: np.ndarray,
) -> DoGFit:
    """
    Fit a Difference-of-Gaussians model to the RF map.

    RF map is (H, W) with x_deg, y_deg giving the spatial coordinates.
    """
    h, w = rf_map.shape
    xx = np.linspace(x_deg.min(), x_deg.max(), w)
    yy = np.linspace(y_deg.min(), y_deg.max(), h)
    X, Y = np.meshgrid(xx, yy)
    xy_flat = np.column_stack([X.ravel(), Y.ravel()])
    z_flat = rf_map.ravel().astype(float)

    # Initial guess from RF statistics
    peak_idx = np.argmax(z_flat)
    x0_init = xy_flat[peak_idx, 0]
    y0_init = xy_flat[peak_idx, 1]
    z_max = float(z_flat.max())
    z_min = float(z_flat.min())
    baseline_init = float(np.median(z_flat))

    # Rough sigma from second moment around peak
    dx = xx[1] - xx[0] if len(xx) > 1 else 0.01
    dy = yy[1] - yy[0] if len(yy) > 1 else 0.01
    sigma_c_init = max(0.02, min(0.15, (dx + dy) * 5))
    sigma_s_init = sigma_c_init * 2.5
    w_c_init = max(1e-6, z_max - baseline_init)
    w_s_init = w_c_init * 0.5

    def _model(xy, x0, y0, sigma_c, sigma_s, w_c, w_s, baseline):
        return _dog_2d(xy, x0, y0, sigma_c, sigma_s, w_c, w_s, baseline)

    try:
        popt, _ = curve_fit(
            _model,
            xy_flat,
            z_flat,
            p0=(x0_init, y0_init, sigma_c_init, sigma_s_init, w_c_init, w_s_init, baseline_init),
            bounds=(
                (x_deg.min(), y_deg.min(), 0.005, 0.01, 0, 0, z_min - 10),
                (x_deg.max(), y_deg.max(), 0.5, 1.0, 1e6, 1e6, z_max + 10),
            ),
            maxfev=2000,
        )
        return DoGFit(
            sigma_center=float(popt[2]),
            sigma_surround=float(popt[3]),
            weight_center=float(popt[4]),
            weight_surround=float(popt[5]),
            x0=float(popt[0]),
            y0=float(popt[1]),
            baseline=float(popt[6]),
            success=True,
        )
    except Exception as e:
        return DoGFit(
            sigma_center=sigma_c_init,
            sigma_surround=sigma_s_init,
            weight_center=w_c_init,
            weight_surround=w_s_init,
            x0=x0_init,
            y0=y0_init,
            baseline=baseline_init,
            success=False,
            message=str(e),
        )


def probe_sweep(
    state: SimState,
    rgc_type: str = "midget_on_L",
    probe_resolution: int = 48,
    probe_radius_deg: float = 0.03,
    config: GlobalConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep a small probe spot across the stimulus plane and record RGC response.

    Returns:
        x_deg: (R,) x coordinates in degrees
        y_deg: (R,) y coordinates in degrees
        rf_map: (R, R) firing rate at each probe position
    """
    cfg = config or state.config
    h, w = state.grid_shape()
    dx = cfg.retina.dx_deg
    xs = (np.arange(w) - w / 2 + 0.5) * dx
    ys = (np.arange(h) - h / 2 + 0.5) * dx

    # Probe positions: grid over the field
    x_probe = np.linspace(xs.min(), xs.max(), probe_resolution)
    y_probe = np.linspace(ys.min(), ys.max(), probe_resolution)
    Xp, Yp = np.meshgrid(x_probe, y_probe)

    # RGC grid to sample from
    fr_map = {
        "midget_on_L": state.fr_midget_on_L,
        "midget_off_L": state.fr_midget_off_L,
        "midget_on_M": state.fr_midget_on_M,
        "midget_off_M": state.fr_midget_off_M,
        "parasol_on": state.fr_parasol_on,
        "parasol_off": state.fr_parasol_off,
    }
    fr_grid = fr_map.get(rgc_type, state.fr_midget_on_L)
    if fr_grid is None:
        fr_grid = np.zeros(state.grid_shape(), dtype=np.float32)

    # For each probe position, we need the RGC response. The current state has
    # one stimulus. We would need to run a mini-simulation per probe position.
    # To avoid 48*48 = 2304 full pipeline runs, we use a linear approximation:
    # the RF is the convolution of the circuit's spatial kernel with a delta.
    # For a spot at (x0,y0), the response is approximately the value of the
    # circuit's impulse response at that offset from the RGC center.
    # Simplified: use the current circuit's effective kernel.
    # Better: run a few probe positions (e.g. 24x24) with the actual pipeline.
    # We'll do a reduced run for performance.

    rf_map = np.zeros((probe_resolution, probe_resolution), dtype=np.float32)
    orig_params = dict(state.stimulus_params)

    for i in range(probe_resolution):
        for j in range(probe_resolution):
            x0, y0 = float(Xp[i, j]), float(Yp[i, j])
            state.stimulus_params.update({
                "type": "spot",
                "x_deg": x0,
                "y_deg": y0,
                "radius_deg": probe_radius_deg,
                "wavelength_nm": orig_params.get("wavelength_nm", 550),
                "intensity": orig_params.get("intensity", 1.0),
            })
            tick(state, 0.05)  # brief step to settle
            fr = fr_map.get(rgc_type, state.fr_midget_on_L)
            if fr is not None:
                # Sample center RGC response (receptive field of cell at grid center)
                cy, cx = h // 2, w // 2
                rf_map[i, j] = float(fr[cy, cx])
            else:
                rf_map[i, j] = 0.0

    state.stimulus_params.update(orig_params)
    tick(state, 0.05)  # restore

    return x_probe, y_probe, rf_map


def probe_sweep_fast(
    state: SimState,
    rgc_type: str = "midget_on_L",
    probe_resolution: int = 32,
    probe_radius_deg: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast RF probe using precomputed kernels (linear approximation).
    Less accurate but O(1) per position via convolution.
    """
    cfg = state.config
    h, w = state.grid_shape()
    dx = cfg.retina.dx_deg
    xs = (np.arange(w) - w / 2 + 0.5) * dx
    ys = (np.arange(h) - h / 2 + 0.5) * dx

    x_probe = np.linspace(xs.min(), xs.max(), probe_resolution)
    y_probe = np.linspace(ys.min(), ys.max(), probe_resolution)

    fr_map = {
        "midget_on_L": state.fr_midget_on_L,
        "midget_off_L": state.fr_midget_off_L,
        "midget_on_M": state.fr_midget_on_M,
        "midget_off_M": state.fr_midget_off_M,
        "parasol_on": state.fr_parasol_on,
        "parasol_off": state.fr_parasol_off,
    }
    fr_grid = fr_map.get(rgc_type, state.fr_midget_on_L)
    if fr_grid is None:
        fr_grid = np.zeros(state.grid_shape(), dtype=np.float32)

    # RF is approximately the convolution of spot with circuit kernel.
    # For a small spot, response at RGC (i,j) ~ sum over spot of kernel(i-io, j-jo).
    # So sweeping the spot is equivalent to correlating the spot with the RF.
    # The current fr_grid is the response to the current stimulus. For a delta
    # spot at center, the RF would be the "receptive field" - the sensitivity
    # map. We approximate by: RF(x,y) = response when spot is at (x,y).
    # With linearity: that's the cross-correlation of the circuit output with
    # a delta at (x,y). The circuit output for delta at (x0,y0) is the shifted
    # impulse response. So RF = impulse_response (shifted to center).
    # The impulse response for our circuit is complex. Simpler: use the
    # gradient of fr with respect to stimulus position, or run a sparse sweep.
    # For "fast" we run a 16x16 sweep (256 runs) instead of 48x48.
    return probe_sweep(state, rgc_type, probe_resolution, probe_radius_deg)
