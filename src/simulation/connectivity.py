"""
Lazy connectivity computation for the picked cell.

Computes input tree (what drives this cell) and output (what it drives).
Caches last N picks for instant re-display. Uses cKDTree for O(log N) radius queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.simulation.bio_constants import (
    CONE_FRAC_L,
    CONE_FRAC_M,
    CONE_FRAC_S,
    MICRONS_PER_PX,
)
from src.simulation.cell_positions import (
    CONE_TYPE_L,
    CONE_TYPE_M,
    CONE_TYPE_S,
    CellPositions,
)
from src.simulation.scale import (
    get_eccentricity_deg,
    get_midget_dend_radius_um,
    get_parasol_dend_radius_um,
)

# Radii in μm for connectivity (from literature)
BIPOLAR_DEND_RADIUS_UM = 15.0  # midget/diffuse dendritic field
ROD_BIPOLAR_POOLING_UM = 30.0
AII_AMACRINE_RADIUS_UM = 50.0
WIDE_FIELD_AMACRINE_RADIUS_UM = 500.0
HORIZONTAL_POOLING_UM = 200.0

CACHE_SIZE = 10


@dataclass
class ConeBreakdown:
    """Cone input counts for L/M/S and ratio check vs biological norm."""
    n_L: int = 0
    n_M: int = 0
    n_S: int = 0

    @property
    def total(self) -> int:
        return self.n_L + self.n_M + self.n_S

    @property
    def ratio_L(self) -> float:
        return self.n_L / max(1, self.total)

    @property
    def ratio_M(self) -> float:
        return self.n_M / max(1, self.total)

    @property
    def ratio_S(self) -> float:
        return self.n_S / max(1, self.total)

    def deviation_from_norm(self) -> Tuple[float, float, float]:
        """Difference from CONE_FRAC_L/M/S (64/32/2%)."""
        return (
            self.ratio_L - CONE_FRAC_L,
            self.ratio_M - CONE_FRAC_M,
            self.ratio_S - CONE_FRAC_S,
        )


@dataclass
class RGCConnectivityResult:
    """Full input/output tree for a picked RGC."""
    cell_id: int
    position_px: Tuple[float, float]
    position_um: Tuple[float, float]
    eccentricity_deg: float
    dendritic_diameter_um: float
    pathway: str  # "Magnocellular (M)", "Parvocellular (P)", "Koniocellular (K)"
    firing_rate: float

    # Inputs
    bipolar_on_midget: int = 0
    bipolar_on_diffuse: int = 0
    bipolar_off_midget: int = 0
    bipolar_off_diffuse: int = 0
    cone_breakdown: ConeBreakdown = field(default_factory=ConeBreakdown)
    rod_count: int = 0
    amacrine_aii: int = 0
    amacrine_wide: int = 0

    # Convergence summary
    total_photoreceptor_inputs: int = 0
    biological_expectation_range: Tuple[int, int] = (300, 500)


@dataclass
class ConeConnectivityResult:
    """Connectivity for a picked cone."""
    cell_id: int
    position_px: Tuple[float, float]
    cone_type: str  # "L", "M", "S"
    eccentricity_deg: float
    bipolar_count: int = 0
    horizontal_count: int = 0
    rgc_count_upstream: int = 0
    rgc_ids: List[int] = field(default_factory=list)  # RGC indices this cone feeds (up to max listed)


@dataclass
class BipolarConnectivityResult:
    """Connectivity for a picked bipolar."""
    cell_id: int
    position_px: Tuple[float, float]
    cell_type: str  # "ON midget", "OFF diffuse", ...
    cone_count: int = 0
    amacrine_count: int = 0
    rgc_count: int = 0
    activation: float = 0.0


@dataclass
class HorizontalConnectivityResult:
    """Connectivity for a picked horizontal cell."""
    cell_id: int
    position_px: Tuple[float, float]
    dendritic_extent_um: float
    cone_count: int = 0
    cone_types: Optional[ConeBreakdown] = None


@dataclass
class AmacrineConnectivityResult:
    """Connectivity for a picked amacrine cell."""
    cell_id: int
    position_px: Tuple[float, float]
    cell_type: str  # "AII", "wide-field", ...
    reach_um: float
    rgc_count_inhibited: int = 0


def _radius_um_to_px(radius_um: float, microns_per_px: float) -> float:
    return radius_um / microns_per_px


def compute_rgc_connectivity(
    cp: CellPositions,
    rgc_idx: int,
    fovea_center: Tuple[float, float],
    firing_rate: float = 0.0,
) -> RGCConnectivityResult:
    """Compute full input tree for the RGC at rgc_idx. Lazy, on-demand."""
    pos = cp.rgc_positions[rgc_idx]
    x_px, y_px = float(pos[0]), float(pos[1])
    x_um = x_px * cp.microns_per_px
    y_um = y_px * cp.microns_per_px
    ecc = get_eccentricity_deg(x_px, y_px, fovea_center[0], fovea_center[1], cp.microns_per_px)
    dend_radius_um = float(cp.rgc_dendritic_radius_px[rgc_idx] * cp.microns_per_px)
    dend_diameter_um = dend_radius_um * 2.0
    dend_radius_px = dend_radius_um / cp.microns_per_px

    pathway = "Magnocellular (M)  →  LGN layer 1/2"  # parasol default

    # Bipolars within dendritic field
    bipolar_indices: List[int] = []
    if cp._bipolar_tree is not None:
        bipolar_indices = cp._bipolar_tree.query_ball_point([x_px, y_px], r=dend_radius_px)
    n_bipolar_midget_on = min(23, max(0, len(bipolar_indices) // 2))  # placeholder split
    n_bipolar_diffuse_on = min(8, max(0, len(bipolar_indices) - n_bipolar_midget_on))

    # Cones within bipolar dendritic radius of each bipolar (approximate: cones within RGC dend field + bipolar radius)
    cone_radius_px = dend_radius_px + _radius_um_to_px(BIPOLAR_DEND_RADIUS_UM, cp.microns_per_px)
    cone_indices: List[int] = []
    if cp._cone_tree is not None:
        cone_indices = cp._cone_tree.query_ball_point([x_px, y_px], r=cone_radius_px)
    cone_types = cp.cone_types
    n_L = int(np.sum(cone_types[cone_indices] == CONE_TYPE_L)) if len(cone_indices) else 0
    n_M = int(np.sum(cone_types[cone_indices] == CONE_TYPE_M)) if len(cone_indices) else 0
    n_S = int(np.sum(cone_types[cone_indices] == CONE_TYPE_S)) if len(cone_indices) else 0
    cone_breakdown = ConeBreakdown(n_L=n_L, n_M=n_M, n_S=n_S)

    # Rods within rod-bipolar pooling
    rod_radius_px = _radius_um_to_px(ROD_BIPOLAR_POOLING_UM, cp.microns_per_px)
    rod_indices: List[int] = []
    if cp._rod_tree is not None:
        rod_indices = cp._rod_tree.query_ball_point([x_px, y_px], r=rod_radius_px)
    n_rods = len(rod_indices)

    # Amacrines (inhibitory)
    aii_px = _radius_um_to_px(AII_AMACRINE_RADIUS_UM, cp.microns_per_px)
    wide_px = _radius_um_to_px(WIDE_FIELD_AMACRINE_RADIUS_UM, cp.microns_per_px)
    aii_indices = cp._amacrine_tree.query_ball_point([x_px, y_px], r=aii_px) if cp._amacrine_tree else []
    wide_indices = cp._amacrine_tree.query_ball_point([x_px, y_px], r=wide_px) if cp._amacrine_tree else []
    n_aii = len(aii_indices)
    n_wide = len([i for i in wide_indices if i not in set(aii_indices)])

    total_photo = cone_breakdown.total + n_rods
    expect_lo, expect_hi = 300, 500  # biological at ~8°
    if ecc > 5:
        expect_lo, expect_hi = 300, 500
    elif ecc > 2:
        expect_lo, expect_hi = 200, 400

    return RGCConnectivityResult(
        cell_id=rgc_idx,
        position_px=(x_px, y_px),
        position_um=(x_um, y_um),
        eccentricity_deg=ecc,
        dendritic_diameter_um=dend_diameter_um,
        pathway=pathway,
        firing_rate=firing_rate,
        bipolar_on_midget=n_bipolar_midget_on,
        bipolar_on_diffuse=n_bipolar_diffuse_on,
        bipolar_off_midget=0,
        bipolar_off_diffuse=0,
        cone_breakdown=cone_breakdown,
        rod_count=n_rods,
        amacrine_aii=n_aii,
        amacrine_wide=n_wide,
        total_photoreceptor_inputs=total_photo,
        biological_expectation_range=(expect_lo, expect_hi),
    )


def compute_cone_connectivity(
    cp: CellPositions,
    cone_idx: int,
    fovea_center: Tuple[float, float],
) -> ConeConnectivityResult:
    """Connectivity for a picked cone."""
    pos = cp.cone_positions[cone_idx]
    x_px, y_px = float(pos[0]), float(pos[1])
    ecc = get_eccentricity_deg(x_px, y_px, fovea_center[0], fovea_center[1], cp.microns_per_px)
    t = cp.cone_types[cone_idx]
    cone_type = "L" if t == CONE_TYPE_L else ("M" if t == CONE_TYPE_M else "S")
    bip_radius_px = _radius_um_to_px(BIPOLAR_DEND_RADIUS_UM, cp.microns_per_px)
    bipolar_count = len(cp._bipolar_tree.query_ball_point([x_px, y_px], r=bip_radius_px)) if cp._bipolar_tree else 0
    hc_radius_px = _radius_um_to_px(HORIZONTAL_POOLING_UM, cp.microns_per_px)
    horizontal_count = len(cp._horizontal_tree.query_ball_point([x_px, y_px], r=hc_radius_px)) if cp._horizontal_tree else 0
    rgc_radius_px = _radius_um_to_px(200.0, cp.microns_per_px)  # parasol reach
    rgc_indices = cp._rgc_tree.query_ball_point([x_px, y_px], r=rgc_radius_px) if cp._rgc_tree else []
    rgc_count = len(rgc_indices)
    # Store first 50 RGC IDs so inspector can show "which RGCs"
    rgc_ids = list(rgc_indices[:50]) if rgc_indices else []
    return ConeConnectivityResult(
        cell_id=cone_idx,
        position_px=(x_px, y_px),
        cone_type=cone_type,
        eccentricity_deg=ecc,
        bipolar_count=bipolar_count,
        horizontal_count=horizontal_count,
        rgc_count_upstream=rgc_count,
        rgc_ids=rgc_ids,
    )


def compute_bipolar_connectivity(
    cp: CellPositions,
    bipolar_idx: int,
    activation: float = 0.0,
) -> BipolarConnectivityResult:
    """Connectivity for a picked bipolar cell."""
    pos = cp.bipolar_positions[bipolar_idx]
    x_px, y_px = float(pos[0]), float(pos[1])
    cone_radius_px = _radius_um_to_px(BIPOLAR_DEND_RADIUS_UM, cp.microns_per_px)
    cone_count = len(cp._cone_tree.query_ball_point([x_px, y_px], r=cone_radius_px)) if cp._cone_tree else 0
    amacrine_radius_px = _radius_um_to_px(WIDE_FIELD_AMACRINE_RADIUS_UM, cp.microns_per_px)
    amacrine_count = len(cp._amacrine_tree.query_ball_point([x_px, y_px], r=amacrine_radius_px)) if cp._amacrine_tree else 0
    rgc_radius_px = _radius_um_to_px(150.0, cp.microns_per_px)
    rgc_count = len(cp._rgc_tree.query_ball_point([x_px, y_px], r=rgc_radius_px)) if cp._rgc_tree else 0
    return BipolarConnectivityResult(
        cell_id=bipolar_idx,
        position_px=(x_px, y_px),
        cell_type="ON diffuse",
        cone_count=cone_count,
        amacrine_count=amacrine_count,
        rgc_count=rgc_count,
        activation=activation,
    )


def compute_horizontal_connectivity(
    cp: CellPositions,
    hc_idx: int,
) -> HorizontalConnectivityResult:
    """Connectivity for a picked horizontal cell."""
    pos = cp.horizontal_positions[hc_idx]
    x_px, y_px = float(pos[0]), float(pos[1])
    radius_px = _radius_um_to_px(HORIZONTAL_POOLING_UM, cp.microns_per_px)
    cone_indices = cp._cone_tree.query_ball_point([x_px, y_px], r=radius_px) if cp._cone_tree else []
    cone_types = cp.cone_types
    n_L = int(np.sum(cone_types[cone_indices] == CONE_TYPE_L)) if cone_indices else 0
    n_M = int(np.sum(cone_types[cone_indices] == CONE_TYPE_M)) if cone_indices else 0
    n_S = int(np.sum(cone_types[cone_indices] == CONE_TYPE_S)) if cone_indices else 0
    return HorizontalConnectivityResult(
        cell_id=hc_idx,
        position_px=(x_px, y_px),
        dendritic_extent_um=HORIZONTAL_POOLING_UM,
        cone_count=len(cone_indices),
        cone_types=ConeBreakdown(n_L=n_L, n_M=n_M, n_S=n_S),
    )


def compute_amacrine_connectivity(
    cp: CellPositions,
    amacrine_idx: int,
) -> AmacrineConnectivityResult:
    """Connectivity for a picked amacrine cell."""
    pos = cp.amacrine_positions[amacrine_idx]
    x_px, y_px = float(pos[0]), float(pos[1])
    reach_px = _radius_um_to_px(WIDE_FIELD_AMACRINE_RADIUS_UM, cp.microns_per_px)
    rgc_indices = cp._rgc_tree.query_ball_point([x_px, y_px], r=reach_px) if cp._rgc_tree else []
    return AmacrineConnectivityResult(
        cell_id=amacrine_idx,
        position_px=(x_px, y_px),
        cell_type="wide-field",
        reach_um=WIDE_FIELD_AMACRINE_RADIUS_UM,
        rgc_count_inhibited=len(rgc_indices),
    )


class ConnectivityCache:
    """Cache last N connectivity results keyed by (layer_name, cell_id)."""

    def __init__(self, max_size: int = CACHE_SIZE):
        self._max_size = max_size
        self._order: List[Tuple[str, int]] = []
        self._cache: Dict[Tuple[str, int], Any] = {}

    def get(self, layer_name: str, cell_id: int) -> Optional[Any]:
        return self._cache.get((layer_name, cell_id))

    def put(self, layer_name: str, cell_id: int, result: Any) -> None:
        key = (layer_name, cell_id)
        if key in self._cache:
            self._order.remove(key)
        self._cache[key] = result
        self._order.append(key)
        while len(self._order) > self._max_size:
            old = self._order.pop(0)
            del self._cache[old]
