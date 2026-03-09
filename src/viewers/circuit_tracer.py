"""
Circuit tree resolution for 3D viewer selection mode.

Walks connectivity graph upstream (toward cones) and downstream (toward RGC)
using cKDTree radius queries. Produces connection line geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial import cKDTree

from src.simulation.connectivity import (
    AII_AMACRINE_RADIUS_UM,
    BIPOLAR_DEND_RADIUS_UM,
    HORIZONTAL_POOLING_UM,
    WIDE_FIELD_AMACRINE_RADIUS_UM,
)
from src.simulation.bio_constants import ConnectivityConstants

# Connection types for line coloring
CONN_EXCITATORY = (0.27, 1.0, 0.53, 1.0)  # #44FF88
CONN_INHIBITORY = (1.0, 0.27, 0.27, 1.0)   # #FF4444
CONN_AMACRINE_LATERAL = (0.67, 0.27, 1.0, 1.0)  # #AA44FF


class ConnectionType(Enum):
    EXCITATORY = "excitatory"      # cone→bipolar, bipolar→RGC
    INHIBITORY = "inhibitory"      # horizontal→cone, amacrine→bipolar
    AMACRINE_LATERAL = "amacrine"  # amacrine lateral


@dataclass
class ConnectionSegment:
    """Single line segment between two cells."""
    start_xyz: Tuple[float, float, float]
    end_xyz: Tuple[float, float, float]
    conn_type: ConnectionType


@dataclass
class CircuitTree:
    """
    Full circuit tree for a picked cell: all cell (layer, index) in circuit,
    and connection segments for drawing.
    """

    picked_layer: str
    picked_index: int
    picked_pos_px: Tuple[float, float]

    # Set of (layer_key, flat_index) in circuit
    cells_in_circuit: FrozenSet[Tuple[str, int]] = field(default_factory=frozenset)

    # Line segments for rendering
    segments: List[ConnectionSegment] = field(default_factory=list)

    # For inspector: convergence counts, etc.
    circuit_data: Dict = field(default_factory=dict)


def _radius_um_to_px(radius_um: float, microns_per_px: float) -> float:
    return radius_um / microns_per_px


def _build_tree_for_dense_grid(
    grid_h: int,
    grid_w: int,
    microns_per_px: float,
    picked_layer: str,
    picked_i: int,
    picked_j: int,
    positions_3d: Dict[str, np.ndarray],  # layer_key -> (N, 3) world xyz
    constants: Optional[ConnectivityConstants] = None,
) -> CircuitTree:
    """
    Resolve circuit for dense grid where each (i,j) is a cell.
    positions_3d maps layer_key -> (H*W, 3) xyz in same order as ravel.
    """
    constants = constants or ConnectivityConstants()
    n = grid_h * grid_w
    cx, cy = grid_w / 2.0, grid_h / 2.0
    flat_idx = picked_i * grid_w + picked_j
    x_px = float(picked_j)
    y_px = float(picked_i)
    pt_deg = (
        (x_px - cx) * (1.0 / grid_w) * grid_w,  # simplify: dx_deg * (x_px - cx)
        (y_px - cy) * (1.0 / grid_h) * grid_h,
    )
    dx_deg = 1.0 / grid_w  # approximate
    x_deg = (x_px - cx) * dx_deg
    y_deg = (y_px - cy) * dx_deg
    x_um = x_px * microns_per_px
    y_um = y_px * microns_per_px

    cells: Set[Tuple[str, int]] = set()
    segments: List[ConnectionSegment] = []

    # Radius in pixels for queries
    bip_px = _radius_um_to_px(BIPOLAR_DEND_RADIUS_UM, microns_per_px)
    hc_px = _radius_um_to_px(HORIZONTAL_POOLING_UM, microns_per_px)
    aii_px = _radius_um_to_px(AII_AMACRINE_RADIUS_UM, microns_per_px)
    wide_px = _radius_um_to_px(WIDE_FIELD_AMACRINE_RADIUS_UM, microns_per_px)

    def idx_to_ij(idx: int) -> Tuple[int, int]:
        i, j = idx // grid_w, idx % grid_w
        return i, j

    def ij_to_pos(i: int, j: int) -> Tuple[float, float]:
        return float(j), float(i)

    # Build 2D KD-trees for each layer (use pixel coords)
    grid_pts = np.column_stack([
        np.arange(n) % grid_w,
        np.arange(n) // grid_w,
    ]).astype(np.float64)

    trees: Dict[str, cKDTree] = {}
    for key in positions_3d:
        trees[key] = cKDTree(grid_pts)

    # Always include picked cell
    cells.add((picked_layer, flat_idx))

    pt_2d = np.array([[x_px, y_px]], dtype=np.float64)

    # Walk upstream and downstream based on layer
    if picked_layer in ("cone_L", "cone_M", "cone_S"):
        # Cone: downstream to bipolar, horizontal
        for idx in trees.get("bipolar_on_midget", cKDTree(grid_pts)).query_ball_point(pt_2d, bip_px)[0]:
            cells.add(("bipolar_on_midget", int(idx)))
            if "bipolar_on_midget" in positions_3d and picked_layer in positions_3d:
                start = tuple(positions_3d[picked_layer][flat_idx])
                end = tuple(positions_3d["bipolar_on_midget"][idx])
                segments.append(ConnectionSegment(start, end, ConnectionType.EXCITATORY))
        for idx in trees.get("horizontal", cKDTree(grid_pts)).query_ball_point(pt_2d, hc_px)[0]:
            cells.add(("horizontal", int(idx)))
            if "horizontal" in positions_3d and picked_layer in positions_3d:
                start = tuple(positions_3d[picked_layer][flat_idx])
                end = tuple(positions_3d["horizontal"][idx])
                segments.append(ConnectionSegment(start, end, ConnectionType.EXCITATORY))

    elif picked_layer == "horizontal":
        # Horizontal: downstream to cones (feedback = inhibitory)
        for cone_key in ("cone_L", "cone_M", "cone_S"):
            if cone_key not in trees:
                continue
            for idx in trees[cone_key].query_ball_point(pt_2d, hc_px)[0]:
                cells.add((cone_key, int(idx)))
                if cone_key in positions_3d and "horizontal" in positions_3d:
                    start = tuple(positions_3d["horizontal"][flat_idx])
                    end = tuple(positions_3d[cone_key][idx])
                    segments.append(ConnectionSegment(start, end, ConnectionType.INHIBITORY))

    elif picked_layer.startswith("bipolar"):
        # Bipolar: upstream cones, downstream RGC, lateral amacrine
        for cone_key in ("cone_L", "cone_M", "cone_S"):
            if cone_key not in trees:
                continue
            for idx in trees[cone_key].query_ball_point(pt_2d, bip_px)[0]:
                cells.add((cone_key, int(idx)))
                if cone_key in positions_3d and picked_layer in positions_3d:
                    start = tuple(positions_3d[cone_key][idx])
                    end = tuple(positions_3d[picked_layer][flat_idx])
                    segments.append(ConnectionSegment(start, end, ConnectionType.EXCITATORY))
        for rgc_key in ("rgc_midget_on", "rgc_parasol_on"):
            if rgc_key not in trees:
                continue
            for idx in trees[rgc_key].query_ball_point(pt_2d, bip_px * 1.5)[0]:
                cells.add((rgc_key, int(idx)))
                if rgc_key in positions_3d and picked_layer in positions_3d:
                    start = tuple(positions_3d[picked_layer][flat_idx])
                    end = tuple(positions_3d[rgc_key][idx])
                    segments.append(ConnectionSegment(start, end, ConnectionType.EXCITATORY))
        for am_key in ("amacrine_aii", "amacrine_wide"):
            if am_key not in trees:
                continue
            for idx in trees[am_key].query_ball_point(pt_2d, wide_px)[0]:
                cells.add((am_key, int(idx)))
                if am_key in positions_3d and picked_layer in positions_3d:
                    start = tuple(positions_3d[am_key][idx])
                    end = tuple(positions_3d[picked_layer][flat_idx])
                    segments.append(ConnectionSegment(start, end, ConnectionType.INHIBITORY))

    elif picked_layer.startswith("amacrine"):
        # Amacrine: upstream bipolar, lateral to other amacrines
        lat_deg = constants.wide_field_amacrine_lateral_span_deg
        lat_px = lat_deg / (1.0 / grid_w) if grid_w > 0 else wide_px
        for bip_key in ("bipolar_on_midget", "bipolar_on_diffuse", "bipolar_off_midget", "bipolar_off_diffuse"):
            if bip_key not in trees:
                continue
            for idx in trees[bip_key].query_ball_point(pt_2d, aii_px)[0]:
                cells.add((bip_key, int(idx)))
                if bip_key in positions_3d and picked_layer in positions_3d:
                    start = tuple(positions_3d[bip_key][idx])
                    end = tuple(positions_3d[picked_layer][flat_idx])
                    segments.append(ConnectionSegment(start, end, ConnectionType.INHIBITORY))
        for am_key in ("amacrine_aii", "amacrine_wide"):
            if am_key == picked_layer:
                continue
            if am_key not in trees:
                continue
            for idx in trees[am_key].query_ball_point(pt_2d, lat_px)[0]:
                cells.add((am_key, int(idx)))
                if am_key in positions_3d and picked_layer in positions_3d:
                    start = tuple(positions_3d[am_key][idx])
                    end = tuple(positions_3d[picked_layer][flat_idx])
                    segments.append(ConnectionSegment(start, end, ConnectionType.AMACRINE_LATERAL))

    elif picked_layer.startswith("rgc"):
        # RGC: upstream bipolar, amacrine
        for bip_key in ("bipolar_on_midget", "bipolar_on_diffuse", "bipolar_off_midget", "bipolar_off_diffuse"):
            if bip_key not in trees:
                continue
            for idx in trees[bip_key].query_ball_point(pt_2d, bip_px * 1.5)[0]:
                cells.add((bip_key, int(idx)))
                if bip_key in positions_3d and picked_layer in positions_3d:
                    start = tuple(positions_3d[bip_key][idx])
                    end = tuple(positions_3d[picked_layer][flat_idx])
                    segments.append(ConnectionSegment(start, end, ConnectionType.EXCITATORY))
        for am_key in ("amacrine_aii", "amacrine_wide"):
            if am_key not in trees:
                continue
            for idx in trees[am_key].query_ball_point(pt_2d, aii_px)[0]:
                cells.add((am_key, int(idx)))
                if am_key in positions_3d and picked_layer in positions_3d:
                    start = tuple(positions_3d[am_key][idx])
                    end = tuple(positions_3d[picked_layer][flat_idx])
                    segments.append(ConnectionSegment(start, end, ConnectionType.INHIBITORY))

    return CircuitTree(
        picked_layer=picked_layer,
        picked_index=flat_idx,
        picked_pos_px=(x_px, y_px),
        cells_in_circuit=frozenset(cells),
        segments=segments,
        circuit_data={"cells_count": len(cells), "segments_count": len(segments)},
    )


class CircuitTracer:
    """
    Resolves circuit tree from cell selection. Caches until stimulus changes.
    """

    def __init__(
        self,
        grid_h: int,
        grid_w: int,
        microns_per_px: float,
        constants: Optional[ConnectivityConstants] = None,
    ):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.microns_per_px = microns_per_px
        self.constants = constants or ConnectivityConstants()
        self._cache: Optional[Tuple[Tuple[str, int, int], CircuitTree]] = None

    def resolve(
        self,
        layer: str,
        cell_i: int,
        cell_j: int,
        positions_3d: Dict[str, np.ndarray],
    ) -> CircuitTree:
        """Resolve circuit for cell at (cell_i, cell_j) in layer."""
        key = (layer, cell_i, cell_j)
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        tree = _build_tree_for_dense_grid(
            self.grid_h,
            self.grid_w,
            self.microns_per_px,
            layer,
            cell_i,
            cell_j,
            positions_3d,
            self.constants,
        )
        self._cache = (key, tree)
        return tree

    def invalidate(self) -> None:
        """Clear cache when stimulus/grid changes."""
        self._cache = None
