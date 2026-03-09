from __future__ import annotations

"""
Global biological and simulation configuration for the RGC simulator.

All numerical values are collected here so the rest of the code can stay
readable and so that future citation links can be attached in one place.
Cell-type ratios and convergence constants (rod:cone ~20:1, photoreceptor:RGC
~100:1, INL proportions) live in src.simulation.bio_constants and are used by
both the 2D and 3D viewers for consistent biological accuracy.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


@dataclass
class RetinaGeometry:
    """Geometry of the simulated retinal patch (in visual degrees)."""

    field_size_deg: float = 1.0  # square patch, e.g. 1° × 1°
    grid_resolution: int = 256   # pixels per side for layer grids

    @property
    def dx_deg(self) -> float:
        return self.field_size_deg / self.grid_resolution


@dataclass
class TemporalConstants:
    """Time constants (seconds) for simple RC smoothing at each layer."""

    cone_tau: float = 0.03
    horizontal_tau: float = 0.05
    bipolar_tau: float = 0.04
    amacrine_tau: float = 0.04
    rgc_tau: float = 0.02


@dataclass
class RGCNonlinearity:
    """LN model parameters for RGC firing-rate transform."""

    r_max: float = 120.0
    x_half: float = 0.0
    slope: float = 4.0


@dataclass
class HorizontalCellParams:
    # Horizontal cell pooling covers ~100–300 μm (many cone diameters);
    # sigma scales with ROD_CONE_RATIO * base spacing (bio_constants).
    sigma_lm_deg: float = 0.05
    sigma_s_deg: float = 0.05
    alpha_lm: float = 0.7
    alpha_s: float = 0.7


@dataclass
class BipolarParams:
    # Bipolar dendritic field: midget ~5–10 μm (narrow), diffuse ~20–30 μm (broader).
    sigma_diffuse_deg: float = 0.08


@dataclass
class AmacrineParams:
    # AII (narrow) ~30–50 μm; wide-field ~200–500 μm (Masland & Raviola 1998).
    sigma_aii_deg: float = 0.04
    sigma_wide_deg: float = 0.12
    gamma_aii: float = 0.6
    gamma_wide: float = 0.3


@dataclass
class DendriticFieldParams:
    # RGC dendritic field: midget ~5–10 μm at fovea; parasol ~100–300 μm (Masland 2012).
    sigma_midget_deg: float = 0.03
    sigma_parasol_deg: float = 0.1


@dataclass
class SpectralConfig:
    """Spectral sampling and cone fundamentals."""

    lambda_min: int = 380
    lambda_max: int = 700
    lambda_step: int = 5
    fundamentals_csv: Path = DATA_DIR / "cone_fundamentals.csv"

    wavelengths: np.ndarray = field(init=False)
    sens_L: np.ndarray = field(init=False)
    sens_M: np.ndarray = field(init=False)
    sens_S: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.wavelengths = np.arange(
            self.lambda_min, self.lambda_max + 1, self.lambda_step, dtype=float
        )
        # Load from CSV if present (Stockman & Sharpe 2000), else smooth gaussians.
        if self.fundamentals_csv.exists():
            data = np.loadtxt(self.fundamentals_csv, delimiter=",", skiprows=1)
            csv_wl = data[:, 0]
            csv_L, csv_M, csv_S = data[:, 1], data[:, 2], data[:, 3]
            # Interpolate to match our wavelength grid
            self.sens_L = np.interp(self.wavelengths, csv_wl, csv_L)
            self.sens_M = np.interp(self.wavelengths, csv_wl, csv_M)
            self.sens_S = np.interp(self.wavelengths, csv_wl, csv_S)
        else:
            lam = self.wavelengths
            self.sens_L = np.exp(-0.5 * ((lam - 560.0) / 30.0) ** 2)
            self.sens_M = np.exp(-0.5 * ((lam - 530.0) / 30.0) ** 2)
            self.sens_S = np.exp(-0.5 * ((lam - 420.0) / 25.0) ** 2)
        # Normalize to max 1
        for arr in (self.sens_L, self.sens_M, self.sens_S):
            max_val = float(arr.max())
            if max_val > 0:
                arr /= max_val


@dataclass
class ConnectivityWeights:
    """Editable weights for 3D connectivity display and pipeline scaling."""

    cone_to_horizontal: float = 1.0
    cone_to_bipolar: float = 1.0
    horizontal_to_cone: float = 1.0
    bipolar_to_amacrine: float = 1.0
    amacrine_to_bipolar: float = 1.0
    bipolar_to_rgc: float = 1.0


@dataclass
class GlobalConfig:
    """Top-level configuration object passed around the app."""

    retina: RetinaGeometry = field(default_factory=RetinaGeometry)
    temporal: TemporalConstants = field(default_factory=TemporalConstants)
    cell_subsample: int = 8  # 3D Signal Flow: grid subsample for cell spheres (higher = fewer instances)
    connectivity_weights: ConnectivityWeights = field(default_factory=ConnectivityWeights)
    rgc_nl: RGCNonlinearity = field(default_factory=RGCNonlinearity)
    horizontal: HorizontalCellParams = field(default_factory=HorizontalCellParams)
    bipolar: BipolarParams = field(default_factory=BipolarParams)
    amacrine: AmacrineParams = field(default_factory=AmacrineParams)
    dendritic: DendriticFieldParams = field(default_factory=DendriticFieldParams)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)


def default_config() -> GlobalConfig:
    """Convenience constructor used by the main app."""
    return GlobalConfig()


def layer_z_positions() -> Tuple[float, ...]:
    """Return canonical Z positions for each layer in world space (legacy flat stack)."""
    return (6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0)


def signal_flow_slab_layout() -> Tuple[Tuple[str, float, float], ...]:
    """
    Signal Flow Column: (layer_name, y_top, thickness).
    Y increases downward (RGC at bottom). Thickness proportional to biology.
    Order top-to-bottom: Stimulus, Cones, Horizontal, Bipolar, Amacrine, RGC.
    """
    return (
        ("Stimulus", 5.5, 0.4),   # thin stimulus layer
        ("Cones", 5.0, 0.6),      # photoreceptors - thickest
        ("Horizontal", 4.2, 0.25),  # thinnest
        ("Bipolar", 3.8, 0.45),
        ("Amacrine", 3.2, 0.35),
        ("RGC", 2.6, 0.5),        # ganglion cell layer
    )
