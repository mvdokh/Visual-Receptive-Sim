"""
Layer geometry: Z positions, flat ↔ sphere projection, visibility, reorder.

Biological layer order (bottom = vitreal / RGC, top = scleral / photoreceptor):
  Z 5.0: Cone photoreceptors (L, M, S)
  Z 4.0: Horizontal cells
  Z 3.0: Bipolar cells
  Z 2.0: Amacrine cells
  Z 1.0: RGC
  Z 0.0: Output plane (not rendered)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from src.simulation.bio_constants import MICRONS_PER_DEGREE, UM_PER_WORLD_UNIT

# Layer Z heights (flat/anatomical mode, world units; 1 w.u. ≈ 40 µm).
# Centers derived from OCT normative data:
#   RGC: 0.00, Amacrine: 1.25, Bipolar: 2.10, Horizontal: 2.90, Cone: 3.80.
LAYER_Z = {
    "cone_L": 3.80,
    "cone_M": 3.80,
    "cone_S": 3.80,
    "horizontal": 2.90,
    "bipolar_on_midget": 2.10,
    "bipolar_off_midget": 2.10,
    "bipolar_on_diffuse": 2.10,
    "bipolar_off_diffuse": 2.10,
    "amacrine_aii": 1.25,
    "amacrine_wide": 1.25,
    "rgc_midget_on": 0.00,
    "rgc_midget_off": 0.00,
    "rgc_parasol_on": 0.00,
    "rgc_parasol_off": 0.00,
    # Stimulus slab (visual only; not part of dense grids)
    "stimulus": 5.50,
}

# Coarse-layer slab half-thickness (visual scaffolding only, world units).
LAYER_THICKNESS: Dict[str, float] = {
    "rgc": 0.50,
    "amacrine": 0.47,
    "bipolar": 0.48,
    "horizontal": 0.20,
    "cone": 1.20,
    "rod": 1.00,
    "stimulus": 0.10,
}

# Simplified layer names for display (one per type)
LAYER_ORDER = [
    "cone_L",
    "cone_M",
    "cone_S",
    "horizontal",
    "bipolar",
    "amacrine",
    "rgc",
]

# Map display name -> cell type keys
LAYER_TO_TYPES: Dict[str, List[str]] = {
    "cone_L": ["cone_L"],
    "cone_M": ["cone_M"],
    "cone_S": ["cone_S"],
    "horizontal": ["horizontal"],
    "bipolar": [
        "bipolar_on_midget",
        "bipolar_off_midget",
        "bipolar_on_diffuse",
        "bipolar_off_diffuse",
    ],
    "amacrine": ["amacrine_aii", "amacrine_wide"],
    "rgc": ["rgc_midget_on", "rgc_midget_off", "rgc_parasol_on", "rgc_parasol_off"],
}


@dataclass
class LayerManager:
    """
    Manages layer Z positions, flat vs spherical geometry, visibility, reorder.
    All geometry computed once at init and at grid resize; not per-frame.
    """

    grid_h: int
    grid_w: int
    dx_deg: float  # degrees per pixel
    microns_per_px: float
    layer_spacing: float = 1.0
    spherical_radius_mm: float = 12.0

    # Base geometry (computed once per resize).
    # Flat: X/Y in world units at full resolution (N,)
    _flat_x_wu: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32), init=False)
    _flat_y_wu: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32), init=False)
    # Sphere: unit vectors (N,) so each layer is r * unit_vec
    _sphere_unit_x: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32), init=False)
    _sphere_unit_y: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32), init=False)
    _sphere_unit_z: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32), init=False)

    # Cache for generated (possibly subsampled) positions:
    # key = (layer_key, spherical, subsample) -> (Nsub, 3) float32
    _pos_cache: Dict[Tuple[str, bool, int], np.ndarray] = field(default_factory=dict, init=False)
    # Cache subsample flat indices: key=subsample -> (Nsub,) int64
    _subsample_idx_cache: Dict[int, np.ndarray] = field(default_factory=dict, init=False)

    # Visibility per layer
    visibility: Dict[str, bool] = field(default_factory=lambda: {k: True for k in LAYER_ORDER})

    # Z override: move layer to different slab (visualization only)
    z_override: Dict[str, float] = field(default_factory=dict, init=False)

    # Reorder for draw order (which appears on top)
    draw_order: List[str] = field(default_factory=lambda: list(LAYER_ORDER), init=False)

    def __post_init__(self) -> None:
        self._compute_geometry()

    def _compute_geometry(self) -> None:
        """Compute base geometry for flat and spherical modes."""
        h, w = self.grid_h, self.grid_w
        cx, cy = w / 2.0, h / 2.0

        # Flat: XY in degrees from center; scale to world units
        # 290 µm/° → 1° = 290 µm; 1 w.u. = 40 µm → 1° = 290/40 ≈ 7.25 w.u.
        deg_to_wu = float(MICRONS_PER_DEGREE / UM_PER_WORLD_UNIT)

        yy_deg = (np.arange(h, dtype=np.float32) - np.float32(cy)) * np.float32(self.dx_deg)
        xx_deg = (np.arange(w, dtype=np.float32) - np.float32(cx)) * np.float32(self.dx_deg)
        XX_deg, YY_deg = np.meshgrid(xx_deg, yy_deg)
        self._flat_x_wu = (XX_deg * np.float32(deg_to_wu)).ravel()
        self._flat_y_wu = (YY_deg * np.float32(deg_to_wu)).ravel()

        # Spherical: precompute unit sphere vectors from (theta, phi) in radians.
        radius_mm = self.spherical_radius_mm
        radius_um = radius_mm * 1000.0
        _ = radius_um  # keep for readability; radius computed per layer in get_positions()

        theta = np.deg2rad(XX_deg).ravel()
        phi = np.deg2rad(YY_deg).ravel()
        cos_phi = np.cos(phi).astype(np.float32, copy=False)
        cos_theta = np.cos(theta).astype(np.float32, copy=False)
        sin_theta = np.sin(theta).astype(np.float32, copy=False)
        self._sphere_unit_x = (cos_phi * cos_theta).astype(np.float32, copy=False)
        self._sphere_unit_y = (cos_phi * sin_theta).astype(np.float32, copy=False)
        self._sphere_unit_z = np.sin(phi).astype(np.float32, copy=False)

        # Invalidate caches (geometry changed)
        self._pos_cache.clear()
        self._subsample_idx_cache.clear()

    def _subsample_indices(self, subsample: int) -> np.ndarray:
        if subsample <= 1:
            # Special-case so callers can skip advanced indexing.
            return np.arange(self.grid_h * self.grid_w, dtype=np.int64)
        if subsample in self._subsample_idx_cache:
            return self._subsample_idx_cache[subsample]
        h, w = self.grid_h, self.grid_w
        rows = np.arange(0, h, subsample, dtype=np.int64)
        cols = np.arange(0, w, subsample, dtype=np.int64)
        flat_idx = (rows[:, None] * w + cols[None, :]).ravel()
        self._subsample_idx_cache[subsample] = flat_idx
        return flat_idx

    def _layer_radius_wu(self, layer_key: str) -> float:
        """World-space radius for spherical mode (layer-specific offsets)."""
        radius_um = float(self.spherical_radius_mm) * 1000.0
        radius_wu = radius_um / float(UM_PER_WORLD_UNIT)
        # Each layer gets slightly different radius (cones outermost, RGC innermost)
        layer_radius_offset = {
            "cone_L": 0.0,
            "cone_M": 0.0,
            "cone_S": 0.0,
            "horizontal": -0.04 * radius_wu,
            "bipolar_on_midget": -0.08 * radius_wu,
            "bipolar_off_midget": -0.08 * radius_wu,
            "bipolar_on_diffuse": -0.08 * radius_wu,
            "bipolar_off_diffuse": -0.08 * radius_wu,
            "amacrine_aii": -0.12 * radius_wu,
            "amacrine_wide": -0.12 * radius_wu,
            "rgc_midget_on": -0.16 * radius_wu,
            "rgc_midget_off": -0.16 * radius_wu,
            "rgc_parasol_on": -0.16 * radius_wu,
            "rgc_parasol_off": -0.16 * radius_wu,
        }
        return float(radius_wu + layer_radius_offset.get(layer_key, 0.0))

    def get_positions(
        self,
        layer_key: str,
        spherical: bool = False,
        subsample: int = 1,
    ) -> np.ndarray:
        """Return (N, 3) positions for the layer. subsample=2 takes every 2nd cell."""
        cache_key = (layer_key, bool(spherical), int(max(1, subsample)))
        cached = self._pos_cache.get(cache_key)
        if cached is not None:
            return cached

        subsample = int(max(1, subsample))
        idx = self._subsample_indices(subsample)
        out = np.empty((idx.shape[0], 3), dtype=np.float32)

        if spherical:
            r = np.float32(self._layer_radius_wu(layer_key))
            out[:, 0] = self._sphere_unit_x[idx] * r
            out[:, 1] = self._sphere_unit_y[idx] * r
            out[:, 2] = self._sphere_unit_z[idx] * r
        else:
            out[:, 0] = self._flat_x_wu[idx]
            out[:, 1] = self._flat_y_wu[idx]
            z = self.z_override.get(layer_key, LAYER_Z.get(layer_key, 0.0))
            z = float(z) * float(self.layer_spacing)
            out[:, 2] = np.float32(z)

        self._pos_cache[cache_key] = out
        return out

    def get_positions_for_grid(
        self,
        layer_key: str,
        spherical: bool = False,
        subsample: int = 1,
    ) -> np.ndarray:
        """Like get_positions but returns positions for a (H,W) grid (possibly subsampled)."""
        arr = self.get_positions(layer_key, spherical=spherical, subsample=subsample)
        return arr

    def resize(self, grid_h: int, grid_w: int) -> None:
        """Recompute geometry for new grid size."""
        self.grid_h = grid_h
        self.grid_w = grid_w
        self._compute_geometry()

    def set_z_override(self, layer_key: str, z: float) -> None:
        """Override Z for visualization (e.g. pull amacrine to own slab)."""
        self.z_override[layer_key] = z
        self._compute_geometry()

    def lerp_to_mode(self, t: float, spherical: bool) -> np.ndarray:
        """
        Get interpolated positions for animation.
        t in [0,1]: 0=flat, 1=sphere.
        Returns dict layer_key -> (N,3).
        """
        out = {}
        # This is only used for animation/debug. It intentionally materializes arrays.
        for key in LAYER_Z:
            flat = self.get_positions(key, spherical=False, subsample=1).astype(np.float32, copy=False)
            sphere = self.get_positions(key, spherical=True, subsample=1).astype(np.float32, copy=False)
            tt = float(np.clip(t, 0.0, 1.0))
            if spherical:
                out[key] = (1.0 - tt) * flat + tt * sphere
            else:
                out[key] = tt * flat + (1.0 - tt) * sphere
        return out
