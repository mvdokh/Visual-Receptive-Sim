"""Shared layout dimensions and Dear PyGui format dicts for side panels."""

from __future__ import annotations

from typing import Tuple

# Center viewport letterbox (child_window fill)
VIEWPORT_PANEL_BG_RGB_U8: Tuple[int, int, int] = (16, 16, 16)

# Side columns: left controls vs right stats (same side-panel font size).
LEFT_PANEL_WIDTH = 384
RIGHT_PANEL_WIDTH = 360
# Text wrap inside side panels (leave margin vs child_window width).
SIDE_PANEL_TEXT_WRAP = LEFT_PANEL_WIDTH - 28
# Sliders and numeric fields: avoid stretching to full column width.
SIDE_SLIDER_WIDTH = min(158, max(116, LEFT_PANEL_WIDTH - 132))
# Stimulus block: slightly narrower than other sliders (many controls in one column).
STIM_SLIDER_WIDTH = min(132, max(100, LEFT_PANEL_WIDTH - 168))
SIDE_SPIN_WIDTH = 104
SIDE_SPIN_WIDE = 120
SIDE_COMBO_WIDTH = min(164, LEFT_PANEL_WIDTH - 72)
# Stimulus type list includes long names (e.g. drifting_grating_full); keep a few px wider than SIDE_COMBO.
STIM_COMBO_WIDTH = min(176, LEFT_PANEL_WIDTH - 60)
# Smaller type in side panels; center viewport keeps the default (main) font size.
LEFT_PANEL_FONT_PX = 11
# Scrollable body inside ``slot_left_stack`` (panels mount here, not on the slot group).
LEFT_STACK_SCROLL_TAG = "slot_left_stack_scroll"

# When "2D All Layers" is active, side panels widen so the mosaic keeps about this
# much horizontal letterbox (each side) in the center viewer (~1 in at 96 px/in).
ALL_LAYERS_VIEWER_MARGIN_PX = 96

# Minimum size: all three panels (left + center min + right) must fit
MIN_VIEWPORT_WIDTH = 400
MIN_WINDOW_SIZE: Tuple[int, int] = (
    MIN_VIEWPORT_WIDTH + LEFT_PANEL_WIDTH + RIGHT_PANEL_WIDTH,
    640,
)

# Dear PyGui input_float: show 3 decimals; avoid "->" in labels (font may render as "?")
CONN_F = {"step": 0.001, "format": "%.3f"}
# Format only — use with per-widget step=... (do not merge CONN_F or step is duplicated)
INPUT_FLOAT_FMT = {"format": "%.3f"}

# Center viewport item tags (must stay stable for main loop / texture updates)
VIEWPORT_TEX_TAG = "rgc_viewport_tex"
VIEWPORT_AREA_TAG = "viewport_area"
VIEWPORT_IMAGE_TAG = "viewport_image"
VIEWPORT_AREA_THEME_TAG = "viewport_area_theme"

# Right column section heights (fixed + flex stats area)
RIGHT_RECORDING_PANEL_HEIGHT = 200
RIGHT_RASTER_PANEL_HEIGHT = 260

# Connectivity sliders
CONN_WEIGHT_MIN = -3.0
CONN_WEIGHT_MAX = 3.0

_VIEWPORT_PANEL_BG_F = tuple(c / 255.0 for c in VIEWPORT_PANEL_BG_RGB_U8) + (1.0,)
VIEWPORT_BG_RGBA: Tuple[float, float, float, float] = _VIEWPORT_PANEL_BG_F
ALL_LAYERS_BG_RGBA: Tuple[float, float, float, float] = _VIEWPORT_PANEL_BG_F
ALL_LAYERS_STRIP_RGBA: Tuple[float, float, float, float] = _VIEWPORT_PANEL_BG_F
