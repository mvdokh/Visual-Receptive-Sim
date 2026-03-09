"""
Sparse cell position arrays and spatial index (cKDTree) for large-field simulation.

Used for cell picking and connectivity queries. Build trees once at init;
query_ball_point is O(log N) for "all cells within radius r of (x,y)".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from src.simulation.bio_constants import (
    CONE_FRAC_L,
    CONE_FRAC_M,
    CONE_FRAC_S,
    GRID_SIZE_PX,
    MICRONS_PER_PX,
    MIDGET_RGC_DENSITY_PER_MM2,
    PARASOL_RGC_DENSITY_PER_MM2,
)
from src.simulation.scale import get_eccentricity_deg, get_midget_dend_radius_um, get_parasol_dend_radius_um


# Cone type enum for connectivity
CONE_TYPE_L, CONE_TYPE_M, CONE_TYPE_S = 0, 1, 2


def _jittered_hexagonal_lattice(
    grid_size: int,
    spacing_px: float,
    jitter_frac: float = 0.4,
    fovea_center: Optional[Tuple[float, float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Positions on a jittered hexagonal lattice (approximate) within [0, grid_size)^2."""
    if rng is None:
        rng = np.random.default_rng(42)
    if fovea_center is None:
        fovea_center = (grid_size / 2.0, grid_size / 2.0)
    positions: List[Tuple[float, float]] = []
    dy = spacing_px * np.sqrt(3) / 2
    ny = int(grid_size / dy) + 1
    for iy in range(ny):
        y = iy * dy
        if y >= grid_size:
            continue
        nx = int(grid_size / spacing_px) + 1
        x_offset = (iy % 2) * (spacing_px / 2)
        for ix in range(nx):
            x = ix * spacing_px + x_offset
            if x >= grid_size:
                continue
            jx = x + rng.uniform(-jitter_frac * spacing_px, jitter_frac * spacing_px)
            jy = y + rng.uniform(-jitter_frac * spacing_px, jitter_frac * spacing_px)
            jx = np.clip(jx, 0, grid_size - 1e-6)
            jy = np.clip(jy, 0, grid_size - 1e-6)
            positions.append((jx, jy))
    return np.array(positions, dtype=np.float64) if positions else np.zeros((0, 2), dtype=np.float64)


def _rgc_spacing_px(grid_size: int, microns_per_px: float) -> float:
    """Approximate RGC spacing in pixels at mid-periphery (~10°). Combined parasol+midget ~750/mm²."""
    density_per_mm2 = PARASOL_RGC_DENSITY_PER_MM2 + MIDGET_RGC_DENSITY_PER_MM2
    # 1 mm² = 1e6 μm²; spacing ≈ 1/sqrt(density) mm = 1000/sqrt(density) μm
    spacing_um = 1000.0 / np.sqrt(density_per_mm2)
    return spacing_um / microns_per_px


@dataclass
class CellPositions:
    """
    Sparse cell positions and KD-trees for all layers.
    Photoreceptors: dense grid positions (one per pixel or subsampled).
    RGC/Bipolar/Horizontal/Amacrine: jittered lattice positions.
    """

    grid_size: int
    microns_per_px: float
    fovea_center: Tuple[float, float]

    # (N, 2) positions in pixel coordinates
    cone_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    cone_types: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))  # L=0, M=1, S=2
    rod_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    rgc_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    rgc_cell_type: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))  # 0=midget_on_L, 1=parasol_on, ...
    bipolar_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    horizontal_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    amacrine_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))

    # Dendritic/radius in pixels (per cell or scalar)
    rgc_dendritic_radius_px: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

    # cKDTree for each (built in build_trees())
    _cone_tree: Optional[cKDTree] = None
    _rod_tree: Optional[cKDTree] = None
    _rgc_tree: Optional[cKDTree] = None
    _bipolar_tree: Optional[cKDTree] = None
    _horizontal_tree: Optional[cKDTree] = None
    _amacrine_tree: Optional[cKDTree] = None

    def build_photoreceptor_positions(self, subsample: int = 1) -> None:
        """Fill cone and rod positions from dense grid (optionally subsampled). Cone types by CONE_FRAC_L/M/S."""
        h = w = self.grid_size
        if subsample < 1:
            subsample = 1
        yy = np.arange(0, h, subsample, dtype=np.float64)
        xx = np.arange(0, w, subsample, dtype=np.float64)
        XX, YY = np.meshgrid(xx, yy)
        positions = np.stack([XX.ravel(), YY.ravel()], axis=1)
        n = len(positions)
        self.cone_positions = positions
        # Assign cone type by biological ratio (L/M/S)
        rng = np.random.default_rng(123)
        u = rng.uniform(0, 1, size=n)
        self.cone_types = np.zeros(n, dtype=np.int32)
        self.cone_types[u < CONE_FRAC_L] = CONE_TYPE_L
        self.cone_types[(u >= CONE_FRAC_L) & (u < CONE_FRAC_L + CONE_FRAC_M)] = CONE_TYPE_M
        self.cone_types[u >= CONE_FRAC_L + CONE_FRAC_M] = CONE_TYPE_S
        self.rod_positions = positions.copy()  # same grid for rods (dense)

    def build_rgc_positions(self, spacing_px: Optional[float] = None) -> None:
        """Fill RGC positions with jittered hexagonal lattice. Optional eccentricity-dependent radius."""
        if spacing_px is None:
            spacing_px = _rgc_spacing_px(self.grid_size, self.microns_per_px)
        pos = _jittered_hexagonal_lattice(self.grid_size, spacing_px, jitter_frac=0.4)
        self.rgc_positions = pos
        n = len(pos)
        self.rgc_cell_type = np.zeros(n, dtype=np.int32)  # placeholder: 0 = parasol_on
        # Dendritic radius in px from eccentricity
        fx, fy = self.fovea_center
        radii_um = np.zeros(n, dtype=np.float64)
        for i in range(n):
            ecc = get_eccentricity_deg(
                pos[i, 0], pos[i, 1], fx, fy, self.microns_per_px
            )
            radii_um[i] = get_parasol_dend_radius_um(ecc)  # default parasol; could mix midget
        self.rgc_dendritic_radius_px = radii_um / self.microns_per_px

    def build_bipolar_positions(self, spacing_px: float = 8.0) -> None:
        """Bipolar cells: denser than RGC."""
        self.bipolar_positions = _jittered_hexagonal_lattice(
            self.grid_size, spacing_px, jitter_frac=0.3
        )

    def build_horizontal_positions(self, spacing_px: float = 24.0) -> None:
        """Horizontal cells: sparse."""
        self.horizontal_positions = _jittered_hexagonal_lattice(
            self.grid_size, spacing_px, jitter_frac=0.3
        )

    def build_amacrine_positions(self, spacing_px: float = 12.0) -> None:
        """Amacrine cells: medium density."""
        self.amacrine_positions = _jittered_hexagonal_lattice(
            self.grid_size, spacing_px, jitter_frac=0.35
        )

    def build_trees(self) -> None:
        """Build cKDTree for each position array. Call after filling positions."""
        if len(self.cone_positions) > 0:
            self._cone_tree = cKDTree(self.cone_positions)
        if len(self.rod_positions) > 0:
            self._rod_tree = cKDTree(self.rod_positions)
        if len(self.rgc_positions) > 0:
            self._rgc_tree = cKDTree(self.rgc_positions)
        if len(self.bipolar_positions) > 0:
            self._bipolar_tree = cKDTree(self.bipolar_positions)
        if len(self.horizontal_positions) > 0:
            self._horizontal_tree = cKDTree(self.horizontal_positions)
        if len(self.amacrine_positions) > 0:
            self._amacrine_tree = cKDTree(self.amacrine_positions)

    def init_default(self, cone_subsample: int = 1) -> None:
        """Build all positions and trees with default spacing."""
        self.build_photoreceptor_positions(subsample=cone_subsample)
        self.build_rgc_positions()
        self.build_bipolar_positions()
        self.build_horizontal_positions()
        self.build_amacrine_positions()
        self.build_trees()

    @property
    def cone_tree(self) -> Optional[cKDTree]:
        return self._cone_tree

    @property
    def rod_tree(self) -> Optional[cKDTree]:
        return self._rod_tree

    @property
    def rgc_tree(self) -> Optional[cKDTree]:
        return self._rgc_tree

    @property
    def bipolar_tree(self) -> Optional[cKDTree]:
        return self._bipolar_tree

    @property
    def horizontal_tree(self) -> Optional[cKDTree]:
        return self._horizontal_tree

    @property
    def amacrine_tree(self) -> Optional[cKDTree]:
        return self._amacrine_tree


def pick_cell(
    click_x_px: float,
    click_y_px: float,
    layer_name: str,
    cell_positions: np.ndarray,
    pick_radius_px: float = 10.0,
) -> Optional[int]:
    """
    Find the nearest cell in the given layer within pick_radius_px of the click.
    Returns cell index or None if none in range.
    """
    if cell_positions is None or len(cell_positions) == 0:
        return None
    dists = np.sqrt(
        (cell_positions[:, 0] - click_x_px) ** 2 + (cell_positions[:, 1] - click_y_px) ** 2
    )
    nearest_idx = int(np.argmin(dists))
    if dists[nearest_idx] < pick_radius_px:
        return nearest_idx
    return None


def get_positions_for_layer(cp: CellPositions, layer_name: str) -> Optional[np.ndarray]:
    """Return (N, 2) positions array for the given layer name."""
    name = layer_name.lower()
    if "cone" in name or "photoreceptor" in name:
        return cp.cone_positions
    if "rod" in name:
        return cp.rod_positions
    if "rgc" in name or "ganglion" in name:
        return cp.rgc_positions
    if "bipolar" in name:
        return cp.bipolar_positions
    if "horizontal" in name:
        return cp.horizontal_positions
    if "amacrine" in name:
        return cp.amacrine_positions
    return None


# Layer order for auto-pick: only neurons (exclude Cone/Rod so click picks RGC/bipolar/etc., not photoreceptor)
PICK_LAYER_ORDER = ("RGC", "Bipolar", "Amacrine", "Horizontal")


def pick_nearest_cell_any_layer(
    cp: CellPositions,
    click_x_px: float,
    click_y_px: float,
    pick_radius_px: float = 20.0,
) -> Tuple[Optional[str], Optional[int]]:
    """
    Find the nearest cell across all layers; return (layer_name, cell_id) or (None, None).
    Uses the layer whose nearest cell is closest to the click (within pick_radius_px).
    """
    best_layer: Optional[str] = None
    best_id: Optional[int] = None
    best_d2: float = pick_radius_px * pick_radius_px + 1.0

    for layer_name in PICK_LAYER_ORDER:
        positions = get_positions_for_layer(cp, layer_name)
        if positions is None or len(positions) == 0:
            continue
        dists2 = (positions[:, 0] - click_x_px) ** 2 + (positions[:, 1] - click_y_px) ** 2
        idx = int(np.argmin(dists2))
        d2 = float(dists2[idx])
        if d2 < best_d2:
            best_d2 = d2
            best_layer = layer_name
            best_id = idx

    if best_layer is not None and best_d2 <= pick_radius_px * pick_radius_px:
        return best_layer, best_id
    return None, None
