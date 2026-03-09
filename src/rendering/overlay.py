"""
2D overlay drawing for selected cell: dendritic field, surround, cone/bipolar/amacrine scatter.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from src.simulation.cell_positions import CellPositions
from src.simulation.connectivity import (
    AII_AMACRINE_RADIUS_UM,
    BIPOLAR_DEND_RADIUS_UM,
    ROD_BIPOLAR_POOLING_UM,
    HORIZONTAL_POOLING_UM,
    RGCConnectivityResult,
    WIDE_FIELD_AMACRINE_RADIUS_UM,
)


def _radius_um_to_px(radius_um: float, microns_per_px: float) -> float:
    return radius_um / microns_per_px


def draw_cell_overlay(
    grid_shape: Tuple[int, int],
    cp: Optional[CellPositions],
    rgc_result: Optional[RGCConnectivityResult],
    microns_per_px: float,
) -> np.ndarray:
    """
    Draw overlay for a selected RGC: circles (dendritic, HC surround, amacrine reach)
    and scatter dots (L/M/S cones, rods, bipolars, amacrines). Returns (H, W, 4) float 0–1.
    """
    h, w = grid_shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    if cp is None or rgc_result is None:
        return overlay

    x_px, y_px = rgc_result.position_px
    cx, cy = int(round(x_px)), int(round(y_px))
    dend_radius_px = rgc_result.dendritic_diameter_um / 2.0 / microns_per_px
    hc_radius_px = _radius_um_to_px(HORIZONTAL_POOLING_UM, microns_per_px)
    amacrine_radius_px = _radius_um_to_px(WIDE_FIELD_AMACRINE_RADIUS_UM, microns_per_px)

    # Circle 1: RGC dendritic field (solid, bright)
    _draw_circle(overlay, cx, cy, dend_radius_px, (1.0, 1.0, 0.0, 0.9), solid=True)
    # Circle 2: Horizontal cell surround (dashed → solid for simplicity)
    _draw_circle(overlay, cx, cy, hc_radius_px, (1.0, 1.0, 0.5, 0.5), solid=False)
    # Circle 3: Wide-field amacrine reach
    _draw_circle(overlay, cx, cy, amacrine_radius_px, (1.0, 0.5, 0.0, 0.4), solid=False)

    # Scatter: cones within connectivity radius (L=green, M=red, S=blue)
    cone_radius_px = dend_radius_px + _radius_um_to_px(BIPOLAR_DEND_RADIUS_UM, microns_per_px)
    if cp._cone_tree is not None:
        cone_indices = cp._cone_tree.query_ball_point([x_px, y_px], r=cone_radius_px)
        for idx in cone_indices:
            px, py = int(round(cp.cone_positions[idx, 0])), int(round(cp.cone_positions[idx, 1]))
            if 0 <= px < w and 0 <= py < h:
                t = cp.cone_types[idx]
                if t == 0:  # L
                    overlay[py, px, :] = [0.2, 1.0, 0.2, 0.9]
                elif t == 1:  # M
                    overlay[py, px, :] = [1.0, 0.2, 0.2, 0.9]
                else:  # S
                    overlay[py, px, :] = [0.2, 0.4, 1.0, 0.9]

    # Rods (gray) within rod-bipolar range
    rod_radius_px = _radius_um_to_px(ROD_BIPOLAR_POOLING_UM, microns_per_px)
    if cp._rod_tree is not None:
        rod_indices = cp._rod_tree.query_ball_point([x_px, y_px], r=rod_radius_px)
        for idx in rod_indices[:500]:  # cap for performance
            px, py = int(round(cp.rod_positions[idx, 0])), int(round(cp.rod_positions[idx, 1]))
            if 0 <= px < w and 0 <= py < h:
                overlay[py, px, :] = [0.5, 0.5, 0.5, 0.6]

    # Bipolar positions (white rings)
    if cp._bipolar_tree is not None:
        bip_indices = cp._bipolar_tree.query_ball_point([x_px, y_px], r=dend_radius_px)
        for idx in bip_indices[:80]:
            px, py = int(round(cp.bipolar_positions[idx, 0])), int(round(cp.bipolar_positions[idx, 1]))
            if 0 <= px < w and 0 <= py < h:
                overlay[py, px, :] = [1.0, 1.0, 1.0, 0.85]
                _draw_circle(overlay, px, py, 2.0, (1.0, 1.0, 1.0, 0.9), solid=False)

    # Amacrine (purple rings)
    aii_px = _radius_um_to_px(AII_AMACRINE_RADIUS_UM, microns_per_px)
    if cp._amacrine_tree is not None:
        am_indices = cp._amacrine_tree.query_ball_point([x_px, y_px], r=aii_px)
        for idx in am_indices[:20]:
            px, py = int(round(cp.amacrine_positions[idx, 0])), int(round(cp.amacrine_positions[idx, 1]))
            if 0 <= px < w and 0 <= py < h:
                _draw_circle(overlay, px, py, 3.0, (0.7, 0.0, 0.7, 0.8), solid=False)

    return overlay


def _draw_circle(
    rgba: np.ndarray,
    cx: int,
    cy: int,
    radius: float,
    color: Tuple[float, float, float, float],
    solid: bool,
) -> None:
    """Draw a circle into rgba (H, W, 4). color is (r,g,b,a)."""
    h, w = rgba.shape[0], rgba.shape[1]
    r = max(1, int(round(radius)))
    y0 = max(0, cy - r)
    y1 = min(h, cy + r + 1)
    x0 = max(0, cx - r)
    x1 = min(w, cx + r + 1)
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            if solid:
                if d <= radius:
                    rgba[yy, xx, :] = color
            else:
                if radius - 1.0 <= d <= radius + 1.0:
                    rgba[yy, xx, :] = color
