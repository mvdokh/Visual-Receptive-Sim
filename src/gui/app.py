from __future__ import annotations

"""
Dear PyGui application wiring together:
- Simulation state + pipeline
- 2D viewport (single layer or all-layers mosaic)
- Control / analysis panels
"""

import os
import random
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

# Run simulation on main thread (no background worker) for smoother 60 FPS; set SIM_ON_MAIN_THREAD=1
SIM_ON_MAIN_THREAD = os.environ.get("SIM_ON_MAIN_THREAD", "").strip().lower() in ("1", "true", "yes")
# Tick every Nth frame when on main thread to hit 60 FPS (SIM_TICK_EVERY_N=2 → 30 Hz sim)
SIM_TICK_EVERY_N = max(1, int(os.environ.get("SIM_TICK_EVERY_N", "1")))

# Cap main loop at this FPS to reduce CPU use and keep UI responsive
TARGET_FPS = 60

import dearpygui.dearpygui as dpg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Upscale factor for display (texture is grid_resolution * DISPLAY_SCALE)
# 4 = 1024x1024 for a 256 grid: good balance of sharpness vs speed.
DISPLAY_SCALE = 4

from src.config import default_config
from src.rendering.heatmap import (
    block_average_downsample_rgba,
    draw_scale_bar_rgba,
    grid_to_rgba,
    spectrum_to_stimulus_rgba,
)
from src.gui.panels.data_export import (
    export_screenshot_png,
    export_layer_grids_csv,
    export_layer_grids_npy,
)
from src.simulation import SimState, tick
from src.simulation.session_recording import (
    LoadedSessionRecording,
    SessionRecordingBuffer,
    load_session_recording,
)
from src.simulation.rgc_type_constants import ALL_RGC_TYPE_KEYS, FUNCTIONAL_GROUPS
from src.simulation.bio_constants import (
    RELATIVE_DENSITY,
    PHOTORECEPTOR_RGC_RATIO,
    ROD_CONE_RATIO,
    CONE_FRAC_L,
    CONE_FRAC_M,
    CONE_FRAC_S,
    RGCS_TOTAL,
    RODS_TOTAL,
    CONES_TOTAL,
    INL_FRAC_HORIZONTAL,
    INL_FRAC_BIPOLAR,
    INL_FRAC_AMACRINE,
)

# 2D layer combo: (internal_key, display_label)
LAYER_ITEMS_2D = [
    ("Stimulus", "Stimulus"),
    ("Cones L", "Cone (L)"),
    ("Cones M", "Cone (M)"),
    ("Cones S", "Cone (S)"),
    ("Horizontal", "Horizontal"),
    ("Bipolar ON", "Bipolar"),
    ("Amacrine", "Amacrine"),
    ("RGC Firing (L)", "RGC"),
    ("RGC spikes (L)", "RGC spikes"),
]
LAYER_DISPLAY_TO_KEY = {label: key for key, label in LAYER_ITEMS_2D}
LAYER_KEY_TO_DISPLAY = {key: label for key, label in LAYER_ITEMS_2D}

# Map 2D layer combo key -> RELATIVE_DENSITY key for biological scale (bio_constants)
LAYER_KEY_TO_DENSITY = {
    "Cones L": "cones_L",
    "Cones M": "cones_M",
    "Cones S": "cones_S",
    "Horizontal": "horizontal",
    "Bipolar ON": "bipolar",
    "Amacrine": "amacrine",
    "RGC Firing (L)": "rgc",
    "RGC spikes (L)": "rgc",
}


# 2D All Layers: 4 rows × 3 columns — stimulus (span 3), cones L/M/S, H/Bip/Am, RGC (span 3).
# Each entry: (row index 0..3, col 0..2, col_span 1|3, layer_key)
COMPOSITE_LAYOUT_2D: Tuple[Tuple[int, int, int, str], ...] = (
    (0, 0, 3, "Stimulus"),
    (1, 0, 1, "Cones L"),
    (1, 1, 1, "Cones M"),
    (1, 2, 1, "Cones S"),
    (2, 0, 1, "Horizontal"),
    (2, 1, 1, "Bipolar ON"),
    (2, 2, 1, "Amacrine"),
    (3, 0, 3, "RGC Firing (L)"),
)

ALL_LAYERS_ABBREV_3: dict[str, str] = {
    "Stimulus": "STM",
    "Cones L": "CNL",
    "Cones M": "CNM",
    "Cones S": "CNS",
    "Horizontal": "HOR",
    "Bipolar ON": "BIP",
    "Amacrine": "AMA",
    "RGC Firing (L)": "RGC",
}


def _panel_section_gap() -> None:
    """Visual separation between major blocks (Dear PyGui has no draggable column splitters)."""
    dpg.add_spacer(height=4)
    dpg.add_separator()
    dpg.add_spacer(height=4)


def _set_convergence_note(layer_name: str) -> None:
    """Set convergence overlay text from bio_constants (rod:cone ~20:1, photoreceptor→RGC ~100:1)."""
    if not dpg.does_item_exist("layer_convergence_note"):
        return
    notes = {
        "Stimulus": "Stimulus → Photoreceptors",
        "Cones L": f"Photoreceptors → Bipolar: ~{int(PHOTORECEPTOR_RGC_RATIO)}:1 overall convergence",
        "Cones M": f"Rod:cone ~{int(ROD_CONE_RATIO)}:1  |  Cones → Bipolar ~{int(PHOTORECEPTOR_RGC_RATIO)}:1",
        "Cones S": f"Rods ~{RODS_TOTAL//1_000_000}M : Cones ~{CONES_TOTAL//1_000_000}M  |  20:1",
        "Horizontal": f"Horizontal ~{int(INL_FRAC_HORIZONTAL*100)}% INL  |  sparse lateral",
        "Bipolar ON": f"Bipolar ~{int(INL_FRAC_BIPOLAR*100)}% INL  |  Bipolar → RGC ~100:1",
        "Amacrine": f"Amacrine ~{int(INL_FRAC_AMACRINE*100)}% INL  |  Bipolar → RGC ~100:1",
        "RGC Firing (L)": f"~{RGCS_TOTAL//1_000_000}M RGCs  |  ~{int(PHOTORECEPTOR_RGC_RATIO)}:1 photoreceptor→RGC",
    }
    dpg.set_value("layer_convergence_note", notes.get(layer_name, ""))


LEFT_PANEL_WIDTH = 400
RIGHT_PANEL_WIDTH = 340
# Minimum size: all three panels (left + center min + right) must fit
MIN_VIEWPORT_WIDTH = 400
MIN_WINDOW_SIZE: Tuple[int, int] = (
    MIN_VIEWPORT_WIDTH + LEFT_PANEL_WIDTH + RIGHT_PANEL_WIDTH,
    640,
)

# Center viewport: texture letterbox, empty texture, 2D All Layers canvas, and (via theme) child_window fill
VIEWPORT_PANEL_BG_RGB_U8: Tuple[int, int, int] = (16, 16, 16)
_VIEWPORT_PANEL_BG_F = tuple(c / 255.0 for c in VIEWPORT_PANEL_BG_RGB_U8) + (1.0,)
VIEWPORT_BG_RGBA: Tuple[float, float, float, float] = _VIEWPORT_PANEL_BG_F
ALL_LAYERS_BG_RGBA: Tuple[float, float, float, float] = _VIEWPORT_PANEL_BG_F
ALL_LAYERS_STRIP_RGBA: Tuple[float, float, float, float] = _VIEWPORT_PANEL_BG_F


def _default_window_size() -> Tuple[int, int]:
    """Primary monitor size minus margin for title bar / dock; fallback for headless."""
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        pad_w, pad_h = 16, 72
        return (max(MIN_WINDOW_SIZE[0], w - pad_w), max(MIN_WINDOW_SIZE[1], h - pad_h))
    except Exception:
        return (1920, 1080)


# Default size before maximize; also used as minimum target in the first frames
WINDOW_SIZE: Tuple[int, int] = _default_window_size()
VIEWPORT_TEX_TAG = "rgc_viewport_tex"
VIEWPORT_AREA_TAG = "viewport_area"
VIEWPORT_IMAGE_TAG = "viewport_image"
VIEWPORT_AREA_THEME_TAG = "viewport_area_theme"
DATA_EXPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "exports"

# Shared state for export callbacks, RF compute, and mouse orbit (updated each frame)
_shared: dict = {}

# Max display resolution for 2D viewer (block-average downsampling above this)
MAX_DISPLAY_SIDE = 1024


def _render_stimulus_rgba(state: SimState) -> np.ndarray:
    """Return Stimulus layer as (H, W, 4) float32 RGBA in 0–1."""
    stim_type = state.stimulus_params.get("type", "spot")
    # For image stimuli, show the loaded RGB image directly so the
    # user sees the true pixel colors rather than the spectral
    # centroid approximation used internally for cones.
    if stim_type == "image" and "image_mask" in state.stimulus_params:
        from skimage.transform import resize as sk_resize

        img = np.asarray(state.stimulus_params["image_mask"], dtype=np.float32)
        h, w = state.grid_shape()
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[0] != h or img.shape[1] != w:
            img = sk_resize(
                img,
                (h, w),
                order=1,
                mode="reflect",
                anti_aliasing=True,
                preserve_range=True,
            ).astype(np.float32)
        vmax = float(img.max()) if img.size > 0 else 0.0
        if vmax > 1.0:
            img = img / 255.0
        img = np.clip(img, 0.0, 1.0)
        # Apply global intensity as a simple gain for display.
        gain = float(state.stimulus_params.get("intensity", 1.0))
        rgb = np.clip(img * gain, 0.0, 1.0)
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        rgba[..., :3] = rgb
        rgba[..., 3] = 1.0
        return rgba
    if state.stimulus_spectrum is not None:
        wl = state.config.spectral.wavelengths
        return spectrum_to_stimulus_rgba(state.stimulus_spectrum, wl)
    h, w = state.grid_shape()
    return np.zeros((h, w, 4), dtype=np.float32)


def _get_heatmap_colormap() -> str:
    """Return current heatmap colormap name from UI (firing, biphasic, spectral, diverging)."""
    if dpg.does_item_exist("heatmap_colormap_combo"):
        val = dpg.get_value("heatmap_colormap_combo")
        # Map display label to internal name
        cmap_map = {
            "Firing (amber)": "firing",
            "Biphasic": "biphasic",
            "Spectral": "spectral",
            "Diverging": "diverging",
        }
        return cmap_map.get(val, "firing")
    return "firing"


def _render_layer_rgba(state: SimState, layer_name: str) -> np.ndarray:
    """Return non-stimulus layer as (H, W, 4) float32 RGBA in 0–1."""
    layer_map = {
        "Cones L": (state.cone_L, "firing"),
        "Cones M": (state.cone_M, "firing"),
        "Cones S": (state.cone_S, "firing"),
        "Horizontal": (state.h_activation, "firing"),
        "Bipolar ON": (state.bp_diffuse_on, "firing"),
        "Amacrine": (state.amacrine_aii, "firing"),
        "RGC Firing (L)": (state.fr_midget_on_L, "firing"),
        "RGC spikes (L)": (state.spike_midget_on_L, "firing"),
    }
    layer, _ = layer_map.get(layer_name, (state.fr_midget_on_L, "firing"))
    colormap = _get_heatmap_colormap()
    if layer is None:
        layer = np.zeros(state.grid_shape(), dtype=np.float32)
    # Optional: weight by convergence so signal compression is visible (bio_constants)
    if dpg.does_item_exist("biological_scale_2d") and dpg.get_value("biological_scale_2d"):
        dkey = LAYER_KEY_TO_DENSITY.get(layer_name)
        if dkey and dkey in RELATIVE_DENSITY:
            scale = RELATIVE_DENSITY["rgc"] / RELATIVE_DENSITY[dkey]
            layer = np.clip(layer.astype(np.float32) * scale, 0.0, None)
    # Cones: display as "activity" (dark = more glutamate release, light = less)
    if layer_name in ("Cones L", "Cones M", "Cones S"):
        layer = 1.0 - np.clip(layer.astype(np.float32), 0.0, 1.0)
    return grid_to_rgba(layer, colormap=colormap)


def _grid_to_rgba_absolute_firing(
    grid: np.ndarray,
    global_max: float,
    colormap: str = "firing",
) -> np.ndarray:
    """
    Map a 2D activation grid (H, W) to RGBA using a shared absolute max.

    Normalizes by global_max so that intensity changes across tiles are visible
    in the 2D All Layers composite; then applies the chosen colormap.
    """
    g = grid.astype(np.float32)
    if global_max <= 0.0:
        return np.zeros((*g.shape, 4), dtype=np.float32)
    n = np.clip(g / float(global_max), 0.0, 1.0)
    return grid_to_rgba(n, colormap=colormap)


def _resize_rgba_to_hw(rgba: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize (H,W,4) float RGBA to exact (h,w,4)."""
    if rgba.shape[0] == h and rgba.shape[1] == w:
        return rgba
    rgba_clipped = np.clip(rgba, 0.0, 1.0)
    img_pil = Image.fromarray((rgba_clipped * 255.0).astype(np.uint8), mode="RGBA")
    img_pil = img_pil.resize((w, h), Image.BILINEAR)
    out = np.asarray(img_pil, dtype=np.uint8).astype(np.float32) / 255.0
    return out.astype(np.float32)


def _get_tile_abbrev_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Small font for 3-letter codes in the strip above each heatmap."""
    font = _shared.get("tile_abbrev_font")
    if font is not None:
        return font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 9)
    except Exception:
        font = ImageFont.load_default()
    _shared["tile_abbrev_font"] = font
    return font


def _draw_strip_abbrev(canvas: np.ndarray, abbrev: str, x0: int, y0: int, w: int, h: int) -> None:
    """Strip above the heatmap (not on map pixels); 3-letter code top-right, white text."""
    if w <= 0 or h <= 0 or not abbrev:
        return
    ch, cw = canvas.shape[0], canvas.shape[1]
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(cw, x0 + w), min(ch, y0 + h)
    sw, sh = x1c - x0c, y1c - y0c
    if sw <= 0 or sh <= 0:
        return
    strip = tuple(int(c * 255) for c in ALL_LAYERS_STRIP_RGBA[:3]) + (255,)
    layer = Image.new("RGBA", (sw, sh), strip)
    draw = ImageDraw.Draw(layer)
    font = _get_tile_abbrev_font()
    text_color = (255, 255, 255, 255)
    try:
        bbox = draw.textbbox((0, 0), abbrev, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = len(abbrev) * 5, 9
    rx = max(0, sw - tw - 3)
    ry = max(0, (sh - th) // 2)
    draw.text((rx, ry), abbrev, font=font, fill=text_color)
    fg = np.asarray(layer, dtype=np.float32) / 255.0
    canvas[y0c:y1c, x0c:x1c, :] = fg


def _render_all_layers_composite(state: SimState) -> np.ndarray:
    """
    Assemble all 8 layers into a 4×3 (rows × cols) grid: stimulus and RGC span three
    columns but keep native grid aspect (centered, not stretched). Cones L/M/S and
    H/Bip/Am are one column each. Strips above heatmaps hold 3-letter codes (white text).
    """
    grid_h, grid_w = state.grid_shape()
    label_strip_h = max(12, min(18, int(grid_h * 0.055)))
    row_gap = 2
    comp_w = 3 * grid_w
    comp_h = 4 * grid_h + 4 * label_strip_h + 3 * row_gap

    bg = np.asarray(ALL_LAYERS_BG_RGBA, dtype=np.float32)
    canvas = _shared.get("all_layers_rgba")
    if not isinstance(canvas, np.ndarray) or canvas.shape[:2] != (comp_h, comp_w):
        canvas = np.empty((comp_h, comp_w, 4), dtype=np.float32)
        _shared["all_layers_rgba"] = canvas
    canvas[...] = bg

    cone_L = state.cone_L if state.cone_L is not None else None
    cone_M = state.cone_M if state.cone_M is not None else None
    cone_S = state.cone_S if state.cone_S is not None else None
    cone_max_inverted = 1.0

    def tile_rgba_for(layer_key: str) -> np.ndarray:
        if layer_key == "Stimulus":
            return _render_stimulus_rgba(state)
        if layer_key in ("Cones L", "Cones M", "Cones S"):
            if layer_key == "Cones L":
                grid = np.asarray(cone_L if cone_L is not None else np.zeros(state.grid_shape(), dtype=np.float32), dtype=np.float32).copy()
            elif layer_key == "Cones M":
                grid = np.asarray(cone_M if cone_M is not None else np.zeros(state.grid_shape(), dtype=np.float32), dtype=np.float32).copy()
            else:
                grid = np.asarray(cone_S if cone_S is not None else np.zeros(state.grid_shape(), dtype=np.float32), dtype=np.float32).copy()
            if dpg.does_item_exist("biological_scale_2d") and dpg.get_value("biological_scale_2d"):
                dkey = LAYER_KEY_TO_DENSITY.get(layer_key)
                if dkey and dkey in RELATIVE_DENSITY:
                    scale = RELATIVE_DENSITY["rgc"] / RELATIVE_DENSITY[dkey]
                    grid = np.clip(grid * scale, 0.0, None)
            grid = 1.0 - np.clip(grid, 0.0, 1.0)
            return _grid_to_rgba_absolute_firing(grid, cone_max_inverted, _get_heatmap_colormap())
        return _render_layer_rgba(state, layer_key)

    y_cursor = 0
    for row in range(4):
        y_strip = y_cursor
        y_heat = y_cursor + label_strip_h
        cells = [(c0, cspan, lk) for (r, c0, cspan, lk) in COMPOSITE_LAYOUT_2D if r == row]
        for c0, cspan, layer_key in cells:
            x0 = c0 * grid_w
            tile_w = cspan * grid_w
            raw = tile_rgba_for(layer_key)
            if raw.shape[0] != grid_h or raw.shape[1] != grid_w:
                raw = _resize_rgba_to_hw(raw, grid_h, grid_w)
            abbrev = ALL_LAYERS_ABBREV_3.get(layer_key, layer_key[:3].upper())
            if cspan >= 3:
                x_off = max(0, (tile_w - grid_w) // 2)
                canvas[y_heat : y_heat + grid_h, x0 + x_off : x0 + x_off + grid_w, :] = raw
                _draw_strip_abbrev(canvas, abbrev, x0 + x_off, y_strip, grid_w, label_strip_h)
            else:
                canvas[y_heat : y_heat + grid_h, x0 : x0 + grid_w, :] = raw
                _draw_strip_abbrev(canvas, abbrev, x0, y_strip, grid_w, label_strip_h)
        y_cursor += label_strip_h + grid_h
        if row < 3:
            y_cursor += row_gap

    draw_scale_bar_rgba(
        canvas,
        microns_per_px=state.config.retina.microns_per_px,
        scale_bar_um=float(getattr(state.config.viewer_3d, "scale_bar_um", 100.0)),
        position="bottom_left",
    )
    return canvas


def _resize_rgba_to_display(
    rgba: np.ndarray,
    display_h: int,
    display_w: int,
    letterbox_bg: Optional[Tuple[float, float, float, float]] = None,
) -> np.ndarray:
    """Resize float32 RGBA 0–1 image to the fixed display size using a high-quality filter.

    Preserves aspect ratio by letterboxing into the target texture.
    """
    h, w = rgba.shape[0], rgba.shape[1]
    if h == display_h and w == display_w:
        return rgba
    rgba_clipped = np.clip(rgba, 0.0, 1.0)
    try:
        # Compute uniform scale that fits within the target texture.
        scale = min(display_w / max(w, 1), display_h / max(h, 1))
        target_w = max(1, int(w * scale))
        target_h = max(1, int(h * scale))
        img_pil = Image.fromarray((rgba_clipped * 255.0).astype(np.uint8), mode="RGBA")
        img_pil = img_pil.resize((target_w, target_h), Image.BILINEAR)
        small = np.asarray(img_pil, dtype=np.uint8).astype(np.float32) / 255.0
        if small.shape[2] == 3:
            alpha = np.ones((target_h, target_w, 1), dtype=np.float32)
            small = np.concatenate([small, alpha], axis=-1)
        # Letterbox into full texture size (grey for 2D All Layers, dark blue-gray otherwise).
        bg_color = np.asarray(letterbox_bg or VIEWPORT_BG_RGBA, dtype=np.float32)
        out = np.broadcast_to(bg_color, (display_h, display_w, 4)).copy()
        offset_y = max(0, (display_h - target_h) // 2)
        offset_x = max(0, (display_w - target_w) // 2)
        out[offset_y : offset_y + target_h, offset_x : offset_x + target_w, :] = small
        return out.astype(np.float32)
    except Exception:
        # Fallback: downsample then letterbox (never return a crop without fill — that read as black).
        bg = np.asarray(letterbox_bg or VIEWPORT_BG_RGBA, dtype=np.float32)
        out = np.broadcast_to(bg, (display_h, display_w, 4)).copy()
        step_y = max(1, h // max(display_h, 1))
        step_x = max(1, w // max(display_w, 1))
        resized = rgba_clipped[::step_y, ::step_x]
        th, tw = min(resized.shape[0], display_h), min(resized.shape[1], display_w)
        resized = resized[:th, :tw, :]
        oy = max(0, (display_h - th) // 2)
        ox = max(0, (display_w - tw) // 2)
        out[oy : oy + th, ox : ox + tw, :] = resized
        return out.astype(np.float32)


def _update_stimulus_visibility(stim_type: str, state: SimState | None = None) -> None:
    """Show/hide stimulus controls based on type so only relevant sliders are visible."""
    state = state or _shared.get("state")

    def show(tag: str) -> None:
        if dpg.does_item_exist(tag):
            dpg.show_item(tag)

    def hide(tag: str) -> None:
        if dpg.does_item_exist(tag):
            dpg.hide_item(tag)

    # When switching to moving stimuli, set default velocity so they actually move
    if state and stim_type in ("moving_spot", "moving_bar", "moving_grating"):
        state.stimulus_params.setdefault("vx_deg_s", 0.5)
        state.stimulus_params.setdefault("vy_deg_s", 0.0)
        state.stimulus_params.setdefault("motion_mode", "linear")
        state.stimulus_params.setdefault("motion_period_s", 2.0)
        state.stimulus_params.setdefault("motion_osc_amp_deg", 0.2)
        state.stimulus_params.setdefault("motion_osc_hz", 1.0)
        if dpg.does_item_exist("stim_vx"):
            dpg.set_value("stim_vx", state.stimulus_params["vx_deg_s"])
        if dpg.does_item_exist("stim_vy"):
            dpg.set_value("stim_vy", state.stimulus_params["vy_deg_s"])
        if dpg.does_item_exist("stim_motion_mode"):
            dpg.set_value("stim_motion_mode", state.stimulus_params["motion_mode"])
        if dpg.does_item_exist("stim_motion_period"):
            dpg.set_value("stim_motion_period", state.stimulus_params["motion_period_s"])
        if dpg.does_item_exist("stim_motion_amp"):
            dpg.set_value("stim_motion_amp", state.stimulus_params["motion_osc_amp_deg"])
        if dpg.does_item_exist("stim_motion_hz"):
            dpg.set_value("stim_motion_hz", state.stimulus_params["motion_osc_hz"])

    advanced_tags = [
        "stim_x_deg",
        "stim_y_deg",
        "stim_orientation",
        "stim_width",
        "stim_spatial_freq",
        "stim_phase",
        "stim_inner_radius",
        "stim_vx",
        "stim_vy",
        "stim_radius2",
        "stim_x2_deg",
        "stim_y2_deg",
        "stim_wavelength2",
        "stim_intensity2",
        "stim_motion_mode",
        "stim_motion_period",
        "stim_motion_amp",
        "stim_motion_hz",
    ]
    # Hide everything first
    hide("stim_radius")
    hide("stim_load_image_btn")
    for t in advanced_tags:
        hide(t)

    # Mapping from stimulus type to the controls that make sense
    show_map = {
        "full_field": [],
        "spot": ["stim_radius", "stim_x_deg", "stim_y_deg"],
        "annulus": ["stim_radius", "stim_x_deg", "stim_y_deg", "stim_inner_radius"],
        "bar": ["stim_x_deg", "stim_y_deg", "stim_orientation", "stim_width"],
        "grating": ["stim_x_deg", "stim_y_deg", "stim_orientation", "stim_spatial_freq", "stim_phase"],
        "checkerboard": ["stim_x_deg", "stim_y_deg", "stim_width"],
        "moving_spot": [
            "stim_radius",
            "stim_x_deg",
            "stim_y_deg",
            "stim_vx",
            "stim_vy",
            "stim_motion_mode",
            "stim_motion_period",
            "stim_motion_amp",
            "stim_motion_hz",
        ],
        "moving_bar": [
            "stim_x_deg",
            "stim_y_deg",
            "stim_orientation",
            "stim_width",
            "stim_vx",
            "stim_vy",
            "stim_motion_mode",
            "stim_motion_period",
            "stim_motion_amp",
            "stim_motion_hz",
        ],
        "moving_grating": [
            "stim_x_deg",
            "stim_y_deg",
            "stim_orientation",
            "stim_spatial_freq",
            "stim_phase",
            "stim_vx",
            "stim_vy",
            "stim_motion_mode",
            "stim_motion_period",
            "stim_motion_amp",
            "stim_motion_hz",
        ],
        "expanding_ring": ["stim_radius", "stim_x_deg", "stim_y_deg"],
        "drifting_grating_full": ["stim_orientation", "stim_spatial_freq", "stim_phase", "stim_vx", "stim_vy"],
        "dual_spot": [
            "stim_radius",
            "stim_x_deg",
            "stim_y_deg",
            "stim_radius2",
            "stim_x2_deg",
            "stim_y2_deg",
            "stim_wavelength2",
            "stim_intensity2",
        ],
        "image": ["stim_load_image_btn"],
    }
    tags_to_show = show_map.get(stim_type, show_map["spot"])
    for tag in tags_to_show:
        show(tag)


def _update_view_mode_ui(mode: str) -> None:
    """Show/hide controls depending on active 2D mode."""
    # Layer combo is only meaningful in single-layer 2D heatmap mode.
    if dpg.does_item_exist("layer_combo"):
        if mode == "2D Heatmap":
            dpg.show_item("layer_combo")
        else:
            dpg.hide_item("layer_combo")


def _build_menu_bar() -> None:
    with dpg.menu_bar():
        with dpg.menu(label="File"):
            dpg.add_menu_item(label="Quit", callback=lambda: dpg.stop_dearpygui())
        with dpg.menu(label="Simulation"):
            dpg.add_menu_item(label="Pause / Resume")  # placeholder
        with dpg.menu(label="Help"):
            dpg.add_menu_item(label="About", callback=lambda: dpg.show_item("about_window"))


def _build_left_panel(state: SimState) -> None:
    with dpg.child_window(width=LEFT_PANEL_WIDTH, height=-1, border=True, autosize_x=False):
        dpg.add_text("View")
        dpg.add_combo(
            label="Mode",
            items=["2D Heatmap", "2D All Layers"],
            default_value="2D Heatmap",
            tag="view_mode_combo",
            callback=lambda s, a: _update_view_mode_ui(a),
        )
        dpg.add_combo(
            label="Layer",
            items=[label for _k, label in LAYER_ITEMS_2D],
            default_value=LAYER_KEY_TO_DISPLAY.get("RGC Firing (L)", "RGC Firing (L)"),
            tag="layer_combo",
        )
        # Scale bar length is in microns (um). Keep ASCII "um" to avoid font glyph issues.
        try:
            scale_um = float(state.config.viewer_3d.scale_bar_um)
        except Exception:
            scale_um = 100.0
        dpg.add_text(f"Scale bar: {scale_um:.0f} um", tag="scale_bar_text")
        dpg.add_checkbox(
            label="Biological scale (weight by convergence)",
            default_value=False,
            tag="biological_scale_2d",
        )
        dpg.add_combo(
            label="Heatmap colormap",
            items=["Firing (amber)", "Biphasic", "Spectral", "Diverging"],
            default_value="Firing (amber)",
            tag="heatmap_colormap_combo",
        )
        _panel_section_gap()
        dpg.add_text("Stimulus")
        dpg.add_combo(
            label="Stimulus type",
            items=[
                "spot",
                "full_field",
                "annulus",
                "bar",
                "grating",
                "checkerboard",
                "moving_spot",
                "moving_bar",
                "moving_grating",
                "expanding_ring",
                "drifting_grating_full",
                "dual_spot",
                "image",
            ],
            default_value="spot",
            tag="stimulus_type_combo",
            callback=lambda s, a: (_update_stimulus_visibility(a, state), state.stimulus_params.update({"type": a})),
        )
        dpg.add_slider_float(
            label="Wavelength (nm)",
            min_value=380,
            max_value=700,
            default_value=550,
            callback=lambda s, a: state.stimulus_params.update({"wavelength_nm": a}),
        )
        dpg.add_slider_float(
            label="Intensity",
            min_value=0.0,
            max_value=1.0,
            default_value=1.0,
            callback=lambda s, a: state.stimulus_params.update({"intensity": a}),
        )
        dpg.add_slider_float(
            label="Radius (deg)",
            min_value=0.02,
            max_value=0.5,
            default_value=0.15,
            tag="stim_radius",
            callback=lambda s, a: state.stimulus_params.update({"radius_deg": a}),
        )
        dpg.add_button(
            label="Load image stimulus...",
            width=-1,
            tag="stim_load_image_btn",
            callback=lambda: dpg.show_item("stim_image_dialog"),
        )
        with dpg.tree_node(label="Advanced", default_open=False, tag="stim_advanced_node"):
                dpg.add_slider_float(label="X center (deg)", min_value=-0.5, max_value=0.5, default_value=0.0,
                    tag="stim_x_deg", callback=lambda s, a: state.stimulus_params.update({"x_deg": a}))
                dpg.add_slider_float(label="Y center (deg)", min_value=-0.5, max_value=0.5, default_value=0.0,
                    tag="stim_y_deg", callback=lambda s, a: state.stimulus_params.update({"y_deg": a}))
                dpg.add_slider_float(label="Orientation (deg)", min_value=0.0, max_value=180.0, default_value=0.0,
                    tag="stim_orientation", callback=lambda s, a: state.stimulus_params.update({"orientation_deg": a}))
                dpg.add_slider_float(label="Width (deg)", min_value=0.02, max_value=0.4, default_value=0.1,
                    tag="stim_width", callback=lambda s, a: state.stimulus_params.update({"width_deg": a}))
                dpg.add_slider_float(label="Spatial freq (cpd)", min_value=0.5, max_value=8.0, default_value=2.0,
                    tag="stim_spatial_freq", callback=lambda s, a: state.stimulus_params.update({"spatial_freq_cpd": a}))
                dpg.add_slider_float(label="Phase (deg)", min_value=0.0, max_value=360.0, default_value=0.0,
                    tag="stim_phase", callback=lambda s, a: state.stimulus_params.update({"phase_deg": a}))
                dpg.add_slider_float(label="Inner radius (deg)", min_value=0.01, max_value=0.3, default_value=0.05,
                    tag="stim_inner_radius", callback=lambda s, a: state.stimulus_params.update({"inner_radius_deg": a}))
                dpg.add_slider_float(label="Velocity X (deg/s)", min_value=-2.0, max_value=2.0, default_value=0.0,
                    tag="stim_vx", callback=lambda s, a: state.stimulus_params.update({"vx_deg_s": a}))
                dpg.add_slider_float(label="Velocity Y (deg/s)", min_value=-2.0, max_value=2.0, default_value=0.0,
                    tag="stim_vy", callback=lambda s, a: state.stimulus_params.update({"vy_deg_s": a}))
                dpg.add_combo(
                    label="Motion mode",
                    items=["linear", "loop", "oscillate"],
                    default_value="linear",
                    tag="stim_motion_mode",
                    width=-1,
                    callback=lambda s, a: state.stimulus_params.update({"motion_mode": a}),
                )
                dpg.add_slider_float(
                    label="Loop period (s)",
                    min_value=0.2,
                    max_value=20.0,
                    default_value=2.0,
                    tag="stim_motion_period",
                    callback=lambda s, a: state.stimulus_params.update({"motion_period_s": a}),
                )
                dpg.add_slider_float(
                    label="Oscillate amplitude (deg)",
                    min_value=0.02,
                    max_value=0.5,
                    default_value=0.2,
                    tag="stim_motion_amp",
                    callback=lambda s, a: state.stimulus_params.update({"motion_osc_amp_deg": a}),
                )
                dpg.add_slider_float(
                    label="Oscillate frequency (Hz)",
                    min_value=0.05,
                    max_value=5.0,
                    default_value=1.0,
                    tag="stim_motion_hz",
                    callback=lambda s, a: state.stimulus_params.update({"motion_osc_hz": a}),
                )
                dpg.add_slider_float(label="Secondary radius (deg)", min_value=0.02, max_value=0.5, default_value=0.15,
                    tag="stim_radius2", callback=lambda s, a: state.stimulus_params.update({"radius2_deg": a}))
                dpg.add_slider_float(label="Secondary X (deg)", min_value=-0.5, max_value=0.5, default_value=0.25,
                    tag="stim_x2_deg", callback=lambda s, a: state.stimulus_params.update({"x2_deg": a}))
                dpg.add_slider_float(label="Secondary Y (deg)", min_value=-0.5, max_value=0.5, default_value=0.0,
                    tag="stim_y2_deg", callback=lambda s, a: state.stimulus_params.update({"y2_deg": a}))
                dpg.add_slider_float(label="Secondary wavelength (nm)", min_value=380, max_value=700, default_value=450,
                    tag="stim_wavelength2", callback=lambda s, a: state.stimulus_params.update({"wavelength2_nm": a}))
                dpg.add_slider_float(label="Secondary intensity", min_value=0.0, max_value=1.0, default_value=1.0,
                    tag="stim_intensity2", callback=lambda s, a: state.stimulus_params.update({"intensity2": a}))
        _panel_section_gap()
        dpg.add_text("Circuit tuning")
        _build_connectivity_weights_block(state)
        _panel_section_gap()
        _build_rgc_population_block(state)
        _panel_section_gap()
        dpg.add_text("Spike output")
        dpg.add_checkbox(
            label="Generate spikes from RGC rates (Poisson)",
            default_value=state.config.spike_output.enabled,
            tag="spike_output_enabled",
            callback=lambda s, a: setattr(state.config.spike_output, "enabled", bool(a)),
        )
        dpg.add_checkbox(
            label="Spikes from smoothed rates (else LN instant)",
            default_value=state.config.spike_output.use_smoothed_rates,
            tag="spike_use_smoothed",
            callback=lambda s, a: setattr(state.config.spike_output, "use_smoothed_rates", bool(a)),
        )
        _panel_section_gap()
        dpg.add_text("Cell parameters")
        _build_cell_params_block(state)


_CONN_WEIGHT_MIN = -3.0
_CONN_WEIGHT_MAX = 3.0


def _set_conn_weight(state: SimState, key: str, value: float) -> None:
    if hasattr(state.config, "connectivity_weights"):
        v = max(_CONN_WEIGHT_MIN, min(_CONN_WEIGHT_MAX, float(value)))
        setattr(state.config.connectivity_weights, key, v)


def _set_connectivity_dirty() -> None:
    _shared["connectivity_dirty"] = True


def _set_rgc_pop_dirty() -> None:
    _shared["connectivity_dirty"] = True


def _rpc_set_group_scale(state: SimState, group: str, value: float) -> None:
    state.config.rgc_population.group_scales[group] = max(0.0, min(5.0, float(value)))


def _rpc_set_type_weight(state: SimState, type_name: str, value: float) -> None:
    state.config.rgc_population.type_weight_multipliers[type_name] = max(0.0, min(4.0, float(value)))


def _rpc_reset_type_weights(state: SimState) -> None:
    rpc = state.config.rgc_population
    for tn in ALL_RGC_TYPE_KEYS:
        rpc.type_weight_multipliers[tn] = 1.0
        tag = f"rpc_tw_{tn}"
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, 1.0)


def _rpc_reset_group_scales(state: SimState) -> None:
    rpc = state.config.rgc_population
    for g in FUNCTIONAL_GROUPS:
        rpc.group_scales[g] = 1.0
        tag = f"rpc_grp_{g}"
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, 1.0)


def _build_rgc_population_block(state: SimState) -> None:
    rpc = state.config.rgc_population
    dpg.add_text("RGC population (42-type composition; needs enable)")
    dpg.add_checkbox(
        label="Enable population modulation",
        default_value=rpc.enabled,
        tag="rpc_enabled",
        callback=lambda s, a: (setattr(rpc, "enabled", bool(a)), _set_rgc_pop_dirty()),
    )
    dpg.add_text("Group scales reweight all types in each functional group.")
    for g in FUNCTIONAL_GROUPS:
        tag = f"rpc_grp_{g}"
        dpg.add_slider_float(
            label=g.replace("_", " "),
            min_value=0.0,
            max_value=5.0,
            default_value=float(rpc.group_scales.get(g, 1.0)),
            width=-1,
            tag=tag,
            callback=lambda s, a, gg=g: (_rpc_set_group_scale(state, gg, a), _set_rgc_pop_dirty()),
        )
    dpg.add_button(
        label="Reset group scales to 1.0",
        width=-1,
        callback=lambda: (_rpc_reset_group_scales(state), _set_rgc_pop_dirty()),
    )
    with dpg.tree_node(label="Per-type weight multipliers", default_open=False):
        dpg.add_button(
            label="Reset all type weights to 1.0",
            width=-1,
            callback=lambda: (_rpc_reset_type_weights(state), _set_rgc_pop_dirty()),
        )
        with dpg.child_window(height=240, border=True, horizontal_scrollbar=True):
            for tn in ALL_RGC_TYPE_KEYS:
                tag = f"rpc_tw_{tn}"
                dpg.add_input_float(
                    label=tn,
                    default_value=float(rpc.type_weight_multipliers.get(tn, 1.0)),
                    min_value=0.0,
                    max_value=4.0,
                    min_clamped=True,
                    max_clamped=True,
                    width=200,
                    tag=tag,
                    **_CONN_F,
                    callback=lambda s, a, name=tn: (_rpc_set_type_weight(state, name, a), _set_rgc_pop_dirty()),
                )


def _reset_connectivity_weights(state: SimState) -> None:
    cw = state.config.connectivity_weights
    for key in ("cone_to_horizontal", "cone_to_bipolar", "horizontal_to_cone",
                "bipolar_to_amacrine", "amacrine_to_bipolar", "bipolar_to_rgc"):
        setattr(cw, key, 1.0)
    for tag, key in [
        ("conn_cone_to_horizontal", "cone_to_horizontal"),
        ("conn_cone_to_bipolar", "cone_to_bipolar"),
        ("conn_horizontal_to_cone", "horizontal_to_cone"),
        ("conn_bipolar_to_amacrine", "bipolar_to_amacrine"),
        ("conn_amacrine_to_bipolar", "amacrine_to_bipolar"),
        ("conn_bipolar_to_rgc", "bipolar_to_rgc"),
    ]:
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, getattr(cw, key))
    _shared["connectivity_dirty"] = True


# Dear PyGui input_float: show 3 decimals; avoid "->" in labels (font may render as "?")
_CONN_F = {"step": 0.001, "format": "%.3f"}
# Format only — use with per-widget step=... (do not merge _CONN_F or step is duplicated)
_INPUT_FLOAT_FMT = {"format": "%.3f"}


def _build_connectivity_weights_block(state: SimState) -> None:
    """Synaptic weight editors (also used in left panel)."""
    dpg.add_text("Pathway weights (-3 to 3). Negative inverts/scales inhibitory paths; see pipeline.")
    cw = state.config.connectivity_weights
    rows = [
        ("conn_cone_to_horizontal", "Cone to Horizontal", "cone_to_horizontal"),
        ("conn_cone_to_bipolar", "Cone to Bipolar", "cone_to_bipolar"),
        ("conn_horizontal_to_cone", "Horizontal to Cone", "horizontal_to_cone"),
        ("conn_bipolar_to_amacrine", "Bipolar to Amacrine", "bipolar_to_amacrine"),
        ("conn_amacrine_to_bipolar", "Amacrine to Bipolar", "amacrine_to_bipolar"),
        ("conn_bipolar_to_rgc", "Bipolar to RGC", "bipolar_to_rgc"),
    ]
    for tag, label, key in rows:
        dpg.add_input_float(
            label=label,
            default_value=getattr(cw, key),
            min_value=_CONN_WEIGHT_MIN,
            max_value=_CONN_WEIGHT_MAX,
            min_clamped=True,
            max_clamped=True,
            width=140,
            tag=tag,
            **_CONN_F,
            callback=lambda s, a, k=key: (_set_conn_weight(state, k, a), _set_connectivity_dirty()),
        )
    dpg.add_button(
        label="Reset weights to 1.0",
        tag="conn_reset",
        width=-1,
        callback=lambda: _reset_connectivity_weights(state),
    )
    dpg.add_button(
        label="Randomize weights",
        tag="conn_randomize",
        width=-1,
        callback=lambda: _randomize_connectivity_weights(state),
    )


def _build_cell_params_block(state: SimState) -> None:
    """Detailed cell parameters (pathway knobs; not tied to named histological types)."""
    cfg = state.config
    with dpg.tree_node(label="RGC pathway (narrow field)", default_open=True):
        dpg.add_input_float(
            label="Dendritic sigma (deg)",
            default_value=cfg.dendritic.sigma_midget_deg,
            step=0.001,
            **_INPUT_FLOAT_FMT,
            callback=lambda s, a: setattr(cfg.dendritic, "sigma_midget_deg", a),
        )
        dpg.add_input_float(
            label="Max firing (sp/s)",
            default_value=cfg.rgc_nl.r_max,
            step=1.0,
            **_INPUT_FLOAT_FMT,
            callback=lambda s, a: setattr(cfg.rgc_nl, "r_max", a),
        )
        dpg.add_input_float(
            label="LN slope",
            default_value=cfg.rgc_nl.slope,
            step=0.1,
            **_INPUT_FLOAT_FMT,
            callback=lambda s, a: setattr(cfg.rgc_nl, "slope", a),
        )
        dpg.add_input_float(
            label="LN half-point",
            default_value=cfg.rgc_nl.x_half,
            step=0.01,
            **_INPUT_FLOAT_FMT,
            callback=lambda s, a: setattr(cfg.rgc_nl, "x_half", a),
        )
        dpg.add_input_float(
            label="Tau (s)",
            default_value=cfg.temporal.rgc_tau,
            step=0.01,
            **_INPUT_FLOAT_FMT,
            callback=lambda s, a: setattr(cfg.temporal, "rgc_tau", a),
        )
    with dpg.tree_node(label="RGC pathway (wide field)", default_open=False):
        dpg.add_input_float(
            label="Dendritic sigma (deg)",
            default_value=cfg.dendritic.sigma_parasol_deg,
            step=0.001,
            **_INPUT_FLOAT_FMT,
            callback=lambda s, a: setattr(cfg.dendritic, "sigma_parasol_deg", a),
        )
        dpg.add_text("(LN r_max / slope / half-point shared with narrow field above)")
    with dpg.tree_node(label="Bipolar pooling", default_open=False):
        dpg.add_input_float(
            label="Sigma diffuse (deg)",
            default_value=cfg.bipolar.sigma_diffuse_deg,
            step=0.001,
            **_INPUT_FLOAT_FMT,
            callback=lambda s, a: setattr(cfg.bipolar, "sigma_diffuse_deg", a),
        )
        dpg.add_input_float(
            label="Tau (s)",
            default_value=cfg.temporal.bipolar_tau,
            step=0.01,
            **_INPUT_FLOAT_FMT,
            callback=lambda s, a: setattr(cfg.temporal, "bipolar_tau", a),
        )
    with dpg.tree_node(label="Horizontal feedback", default_open=False):
        dpg.add_input_float(
            label="Sigma LM (deg)",
            default_value=cfg.horizontal.sigma_lm_deg,
            step=0.001,
            **_INPUT_FLOAT_FMT,
            callback=lambda s, a: setattr(cfg.horizontal, "sigma_lm_deg", a),
        )
        dpg.add_input_float(
            label="Sigma S (deg)",
            default_value=cfg.horizontal.sigma_s_deg,
            step=0.001,
            **_INPUT_FLOAT_FMT,
            callback=lambda s, a: setattr(cfg.horizontal, "sigma_s_deg", a),
        )
        dpg.add_input_float(
            label="Alpha LM",
            default_value=cfg.horizontal.alpha_lm,
            step=0.01,
            **_INPUT_FLOAT_FMT,
            callback=lambda s, a: setattr(cfg.horizontal, "alpha_lm", a),
        )
        dpg.add_input_float(
            label="Alpha S",
            default_value=cfg.horizontal.alpha_s,
            step=0.01,
            **_INPUT_FLOAT_FMT,
            callback=lambda s, a: setattr(cfg.horizontal, "alpha_s", a),
        )
        dpg.add_input_float(
            label="Tau (s)",
            default_value=cfg.temporal.horizontal_tau,
            step=0.01,
            **_INPUT_FLOAT_FMT,
            callback=lambda s, a: setattr(cfg.temporal, "horizontal_tau", a),
        )
    with dpg.tree_node(label="Lateral inhibition (narrow pool)", default_open=False):
        dpg.add_input_float(
            label="Sigma (deg)",
            default_value=cfg.amacrine.sigma_aii_deg,
            step=0.001,
            **_INPUT_FLOAT_FMT,
            callback=lambda s, a: setattr(cfg.amacrine, "sigma_aii_deg", a),
        )
        dpg.add_input_float(
            label="Gamma (weight)",
            default_value=cfg.amacrine.gamma_aii,
            step=0.01,
            **_INPUT_FLOAT_FMT,
            callback=lambda s, a: setattr(cfg.amacrine, "gamma_aii", a),
        )
        dpg.add_input_float(
            label="Tau (s)",
            default_value=cfg.temporal.amacrine_tau,
            step=0.01,
            **_INPUT_FLOAT_FMT,
            callback=lambda s, a: setattr(cfg.temporal, "amacrine_tau", a),
        )
    with dpg.tree_node(label="Lateral inhibition (wide pool)", default_open=False):
        dpg.add_input_float(
            label="Sigma (deg)",
            default_value=cfg.amacrine.sigma_wide_deg,
            step=0.001,
            **_INPUT_FLOAT_FMT,
            callback=lambda s, a: setattr(cfg.amacrine, "sigma_wide_deg", a),
        )
        dpg.add_input_float(
            label="Gamma (weight)",
            default_value=cfg.amacrine.gamma_wide,
            step=0.01,
            **_INPUT_FLOAT_FMT,
            callback=lambda s, a: setattr(cfg.amacrine, "gamma_wide", a),
        )
        dpg.add_text("(Tau shared with narrow pool above)")
    dpg.add_spacer(height=8)


def _randomize_connectivity_weights(state: SimState) -> None:
    cw = state.config.connectivity_weights
    for key in ("cone_to_horizontal", "cone_to_bipolar", "horizontal_to_cone",
                "bipolar_to_amacrine", "amacrine_to_bipolar", "bipolar_to_rgc"):
        setattr(
            cw,
            key,
            random.uniform(_CONN_WEIGHT_MIN, _CONN_WEIGHT_MAX),
        )
    for tag, key in [
        ("conn_cone_to_horizontal", "cone_to_horizontal"),
        ("conn_cone_to_bipolar", "cone_to_bipolar"),
        ("conn_horizontal_to_cone", "horizontal_to_cone"),
        ("conn_bipolar_to_amacrine", "bipolar_to_amacrine"),
        ("conn_amacrine_to_bipolar", "amacrine_to_bipolar"),
        ("conn_bipolar_to_rgc", "bipolar_to_rgc"),
    ]:
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, getattr(cw, key))
    _shared["connectivity_dirty"] = True


def _build_right_panel(state: SimState) -> None:
    """Right panel: Stats, Plots, Export, Recording."""
    with dpg.child_window(width=RIGHT_PANEL_WIDTH, height=-1, border=True, autosize_x=False):
        with dpg.tab_bar(tag="right_panel_tabs"):
            with dpg.tab(label="Stats"):
                dpg.add_text("Mean FR per RGC type (sp/s)")
                dpg.add_text("", tag="mean_fr_midget_on_L")
                dpg.add_text("", tag="mean_fr_parasol_on")
                dpg.add_spacer(height=4)
                dpg.add_text("L-M and S-(L+M)")
                dpg.add_text("", tag="lm_summary")
                dpg.add_text("", tag="by_summary")
                with dpg.tree_node(label="Per-layer", default_open=False):
                    for name in ["Stimulus", "Cones L", "Cones M", "Horizontal", "Bipolar", "Amacrine", "RGC"]:
                        dpg.add_text("", tag=f"stat_layer_{name}")
                with dpg.tree_node(label="RGC dynamics", default_open=True):
                    dpg.add_text("RGC mean FR (last 100 ticks)", tag="sparkline_label")
                    with dpg.plot(tag="sparkline_plot", height=140, width=-1):
                        dpg.add_plot_axis(dpg.mvXAxis, tag="spark_x")
                        dpg.add_plot_axis(dpg.mvYAxis, tag="spark_y")
                        dpg.add_line_series([], [], tag="sparkline_series", parent="spark_y")
                    dpg.add_text("RGC FR histogram", tag="hist_label")
                    with dpg.plot(tag="hist_plot", height=140, width=-1):
                        dpg.add_plot_axis(dpg.mvXAxis, tag="hist_x")
                        dpg.add_plot_axis(dpg.mvYAxis, tag="hist_y")
                        dpg.add_bar_series([], [], tag="hist_series", parent="hist_y", weight=0.8)
            with dpg.tab(label="Plots"):
                dpg.add_text("Cone mean drive (L / M / S)")
                with dpg.plot(height=170, width=-1, tag="plot_cone_act"):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="", tag="plot_cone_ax")
                    dpg.add_plot_axis(dpg.mvYAxis, label="mean", tag="plot_cone_ay")
                    dpg.add_bar_series(
                        [0, 1, 2],
                        [0.0, 0.0, 0.0],
                        weight=0.45,
                        label="cones",
                        parent="plot_cone_ay",
                        tag="series_cone_bars",
                    )
                dpg.add_text("Opponent means over time (last 80 ticks)")
                with dpg.plot(height=150, width=-1, tag="plot_oppo_ts"):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="tick", tag="plot_oppo_ax")
                    dpg.add_plot_axis(dpg.mvYAxis, label="signal", tag="plot_oppo_ay")
                    dpg.add_line_series([], [], label="L-M", parent="plot_oppo_ay", tag="series_oppo_lm")
                    dpg.add_line_series([], [], label="S-(L+M)", parent="plot_oppo_ay", tag="series_oppo_by")
            with dpg.tab(label="Export"):
                dpg.add_text("Export data")
                dpg.add_button(label="Save screenshot (PNG)", width=-1, tag="btn_export_png", callback=lambda: dpg.show_item("file_dialog_png"))
                dpg.add_button(label="Save layer stats (CSV)", width=-1, tag="btn_export_csv", callback=lambda: dpg.show_item("file_dialog_csv"))
                dpg.add_button(label="Save layer grids (.npy)", width=-1, tag="btn_export_npy", callback=lambda: dpg.show_item("file_dialog_npy"))
            with dpg.tab(label="Recording"):
                dpg.add_text("Capture frames while the sim runs (folder: session_meta.json + session.npz).")
                dpg.add_checkbox(label="Record", tag="rec_enabled", default_value=False)
                dpg.add_input_int(
                    label="Spatial downsample stride",
                    default_value=4,
                    min_value=1,
                    max_value=64,
                    tag="rec_stride",
                )
                dpg.add_button(
                    label="Save session to folder...",
                    width=-1,
                    callback=lambda: dpg.show_item("file_dialog_rec_save"),
                )
                dpg.add_button(
                    label="Load session from folder...",
                    width=-1,
                    callback=lambda: dpg.show_item("file_dialog_rec_load"),
                )
                dpg.add_checkbox(
                    label="Playback (pauses live simulation)",
                    tag="rec_playback",
                    default_value=False,
                )
                dpg.add_slider_int(
                    label="Frame",
                    tag="rec_frame_slider",
                    min_value=0,
                    max_value=0,
                    default_value=0,
                    callback=lambda s, a: _apply_playback_frame(),
                )
                dpg.add_text("", tag="rec_status_text", wrap=400)


def _build_center_viewport(display_width: int, display_height: int) -> None:
    """Center panel: displays the simulation heatmap output."""
    # Match padding around the centered image to RGB(16,16,16) — default ChildBg reads as black.
    r, g, b = VIEWPORT_PANEL_BG_RGB_U8
    with dpg.theme(tag=VIEWPORT_AREA_THEME_TAG):
        with dpg.theme_component(dpg.mvChildWindow):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (r, g, b, 255))
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (r, g, b, 255))
    with dpg.child_window(border=True, width=-1, height=-1, tag=VIEWPORT_AREA_TAG):
        with dpg.group(horizontal=False):
            # Image tag so we can resize/center each frame; initial size placeholder
            dpg.add_image(VIEWPORT_TEX_TAG, tag=VIEWPORT_IMAGE_TAG, width=400, height=400)

    dpg.bind_item_theme(VIEWPORT_AREA_TAG, VIEWPORT_AREA_THEME_TAG)


def _update_stats(state: SimState) -> None:
    if state.fr_midget_on_L is not None:
        mean_on = float(np.mean(state.fr_midget_on_L))
        dpg.set_value("mean_fr_midget_on_L", f"Midget ON (L): {mean_on:5.1f} sp/s")
    if state.fr_parasol_on is not None:
        mean_parasol = float(np.mean(state.fr_parasol_on))
        dpg.set_value("mean_fr_parasol_on", f"Parasol ON: {mean_parasol:5.1f} sp/s")
    if state.lm_opponent is not None:
        dpg.set_value("lm_summary", f"L-M: mean {float(np.mean(state.lm_opponent)):+.3f}")
    if state.by_opponent is not None:
        dpg.set_value("by_summary", f"S-(L+M): mean {float(np.mean(state.by_opponent)):+.3f}")
    # Plots tab: cone means + opponent trajectories
    if state.cone_L is not None and dpg.does_item_exist("series_cone_bars"):
        mL = float(np.mean(state.cone_L))
        mM = float(np.mean(state.cone_M))
        mS = float(np.mean(state.cone_S))
        dpg.set_value("series_cone_bars", [[0, 1, 2], [mL, mM, mS]])
        ymax = max(1e-9, mL, mM, mS) * 1.15
        if dpg.does_item_exist("plot_cone_ay"):
            dpg.set_axis_limits("plot_cone_ay", 0.0, ymax)
        if dpg.does_item_exist("plot_cone_ax"):
            dpg.set_axis_limits("plot_cone_ax", -0.5, 2.5)
    oh = _shared.get("oppo_hist", [])
    if state.lm_opponent is not None and state.by_opponent is not None:
        oh.append((float(np.mean(state.lm_opponent)), float(np.mean(state.by_opponent))))
    oh = oh[-80:]
    _shared["oppo_hist"] = oh
    if oh and dpg.does_item_exist("series_oppo_lm"):
        xs = list(range(len(oh)))
        dpg.set_value("series_oppo_lm", [xs, [p[0] for p in oh]])
        dpg.set_value("series_oppo_by", [xs, [p[1] for p in oh]])
        lms = [p[0] for p in oh]
        bys = [p[1] for p in oh]
        lo = min(min(lms), min(bys))
        hi = max(max(lms), max(bys))
        pad = max((hi - lo) * 0.1, 0.05) if hi > lo else 0.1
        if dpg.does_item_exist("plot_oppo_ay"):
            dpg.set_axis_limits("plot_oppo_ay", lo - pad, hi + pad)
        if dpg.does_item_exist("plot_oppo_ax"):
            dpg.set_axis_limits("plot_oppo_ax", 0.0, max(1.0, float(len(oh) - 1)))
    # Per-layer stats
    layer_data = {
        "Stimulus": np.sum(state.stimulus_spectrum, axis=-1) if state.stimulus_spectrum is not None else None,
        "Cones L": state.cone_L, "Cones M": state.cone_M, "Cones S": state.cone_S,
        "Horizontal": state.h_activation, "Bipolar": state.bp_diffuse_on,
        "Amacrine": state.amacrine_aii, "RGC": state.fr_midget_on_L,
    }
    for name, arr in layer_data.items():
        if arr is not None and dpg.does_item_exist(f"stat_layer_{name}"):
            m, s, mn, mx = float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr))
            dpg.set_value(f"stat_layer_{name}", f"{name}: mean={m:.3f} std={s:.3f} min={mn:.3f} max={mx:.3f}")
    # RGC sparkline (last 100 ticks)
    if state.fr_midget_on_L is not None:
        hist = _shared.get("rgc_fr_history", [])
        hist.append(float(np.mean(state.fr_midget_on_L)))
        hist = hist[-100:]
        _shared["rgc_fr_history"] = hist
        if hist and dpg.does_item_exist("sparkline_series"):
            xs = list(range(len(hist)))
            dpg.set_value("sparkline_series", [xs, hist])
            if dpg.does_item_exist("spark_x") and dpg.does_item_exist("spark_y"):
                mn, mx = min(hist), max(hist)
                pad = max((mx - mn) * 0.1, 1.0) if mx > mn else 1.0
                dpg.set_axis_limits("spark_x", 0.0, max(1, len(hist) - 1))
                dpg.set_axis_limits("spark_y", max(0, mn - pad), mx + pad)
    # RGC histogram (throttled and subsampled to avoid slowdown when RGC is very active)
    _shared["hist_update_tick"] = _shared.get("hist_update_tick", 0) + 1
    if state.fr_midget_on_L is not None and dpg.does_item_exist("hist_series") and (_shared["hist_update_tick"] % 3 == 0):
        flat = state.fr_midget_on_L.flatten()
        flat = flat[np.isfinite(flat)]
        max_hist_points = 2048
        if len(flat) > max_hist_points:
            step = len(flat) // max_hist_points
            flat = flat[::step][:max_hist_points]
        if len(flat) > 0:
            mn, mx = float(np.min(flat)), float(np.max(flat))
            if not np.isfinite(mn):
                mn = 0.0
            if not np.isfinite(mx):
                mx = mn + 1.0
            if mx <= mn:
                mx = mn + 1.0
                bins = 2
            else:
                # NumPy can raise "too many bins" when range is tiny (e.g. 1e-20)
                min_range = max(1e-9, np.finfo(np.float64).tiny * 20)
                if (mx - mn) < min_range:
                    mx = mn + 1.0
                    bins = 2
                else:
                    bins = 16
            try:
                counts, edges = np.histogram(flat, bins=bins, range=(mn, mx))
            except ValueError:
                # Fallback if NumPy still rejects (e.g. version-dependent)
                bins = 2
                mx = mn + 1.0
                counts, edges = np.histogram(flat, bins=bins, range=(mn, mx))
            xs = [(float(edges[i]) + float(edges[i + 1])) / 2 for i in range(bins)]
            counts_list = [float(c) for c in counts]
            dpg.set_value("hist_series", [xs, counts_list])
            if dpg.does_item_exist("hist_x") and dpg.does_item_exist("hist_y"):
                dpg.set_axis_limits("hist_x", float(edges[0]), float(edges[-1]))
                dpg.set_axis_limits("hist_y", 0.0, max(1.0, float(np.max(counts)) * 1.1))


def _load_app_font() -> int | None:
    """Load a modern sans-serif font. Returns font id or None."""
    try:
        import matplotlib.font_manager as fm
        with dpg.font_registry():
            # Modern UI fonts first; fallback to clean system fonts
            for name in ("Inter", "SF Pro Display", "SF Pro Text", "Segoe UI", "Roboto",
                         "Open Sans", "Source Sans 3", "Nunito Sans", "Helvetica Neue", "Arial"):
                path = fm.findfont(name)
                if path and Path(path).exists():
                    return dpg.add_font(path, 14, default_font=True)
    except Exception:
        pass
    return None


def _sim_worker() -> None:
    """Background thread: tick state_back at 60 Hz and swap with state_front. Throttled so UI stays responsive."""
    sim_dt = 1.0 / 60.0
    target_interval = 1.0 / 60.0
    while True:
        try:
            if _shared.get("playback_active"):
                time.sleep(0.016)
                continue
            back = _shared.get("state_back")
            if back is None:
                time.sleep(0.016)
                continue
            t0 = time.perf_counter()
            tick(back, sim_dt)
            front = _shared.get("state_front")
            if front is not None:
                _shared["state_front"], _shared["state_back"] = back, front
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, target_interval - elapsed))
        except Exception:
            time.sleep(0.016)


def _apply_playback_frame() -> None:
    rec = _shared.get("loaded_recording")
    if not isinstance(rec, LoadedSessionRecording) or rec.n_frames <= 0:
        return
    st_front = _shared.get("state_front") or _shared.get("state")
    if st_front is None or not dpg.does_item_exist("rec_frame_slider"):
        return
    idx = int(dpg.get_value("rec_frame_slider"))
    rec.apply_frame(idx, st_front)
    st_back = _shared.get("state_back")
    if st_back is not None and st_back is not st_front:
        rec.apply_frame(idx, st_back)


def run_app() -> None:
    """Create the Dear PyGui + ModernGL app and start the main loop."""
    cfg = default_config()
    state = SimState(config=cfg)
    state.stimulus_params.update({
        "type": "spot",
        "wavelength_nm": 550.0,
        "intensity": 1.0,
        "x_deg": 0.0,
        "y_deg": 0.0,
        "orientation_deg": 0.0,
        "width_deg": 0.1,
        "spatial_freq_cpd": 2.0,
        "phase_deg": 0.0,
        "inner_radius_deg": 0.05,
        "rgb_mapping_mode": "rgbtolms",
    })
    if hasattr(state.config, "spectral"):
        setattr(state.config.spectral, "image_rgb_mapping", "rgbtolms")
    # Second state for background tick (only when using worker); share params so UI updates apply to both
    state_back = None
    if not SIM_ON_MAIN_THREAD:
        state_back = SimState(config=cfg)
        state_back.stimulus_params = state.stimulus_params
        state_back.config = state.config

    dpg.create_context()

    # Load modern font (Inter, SF Pro, Segoe UI, etc.)
    _shared["app_font"] = _load_app_font()

    # Texture registry: cap at MAX_DISPLAY_SIDE for large grids (block-average downsampling)
    grid_h, grid_w = state.grid_shape()
    display_w = min(MAX_DISPLAY_SIDE, grid_w * DISPLAY_SCALE)
    display_h = min(MAX_DISPLAY_SIDE, grid_h * DISPLAY_SCALE)

    with dpg.texture_registry():
        # Initialize with dark gray to avoid flashing on load
        empty_tex = np.broadcast_to(
            np.asarray(VIEWPORT_BG_RGBA, dtype=np.float32),
            (display_h, display_w, 4),
        ).copy()
        dpg.add_dynamic_texture(
            display_w,
            display_h,
            empty_tex.flatten(),
            tag=VIEWPORT_TEX_TAG,
        )

    with dpg.window(
        label="RGC Circuit Simulator",
        tag="main_window",
        width=WINDOW_SIZE[0],
        height=WINDOW_SIZE[1],
        no_title_bar=False,
        no_move=True,
        no_resize=False,
        no_scrollbar=True,
    ):
        _build_menu_bar()

        # Three-column layout: left panel, center viewport, right panel.
        with dpg.group(horizontal=True):
            _build_left_panel(state)
            _build_center_viewport(display_w, display_h)
            _build_right_panel(state)

    _update_stimulus_visibility("spot")  # initial visibility for default type
    _update_view_mode_ui("2D Heatmap")

    # Apply custom font to main window and globally
    app_font = _shared.get("app_font")
    if app_font is not None:
        dpg.bind_font(app_font)
        dpg.bind_item_font("main_window", app_font)

    # About window (hidden by default)
    with dpg.window(label="About", modal=True, show=False, tag="about_window"):
        dpg.add_text("RGC Circuit Simulator — Python")
        dpg.add_text(
            "First-stage human vision simulator: stimulus -> cones -> horizontals -> bipolars "
            "-> amacrines -> RGCs, visualized in layered 2D views."
        )

    dpg.create_viewport(
        title="RGC Circuit Simulator",
        width=WINDOW_SIZE[0],
        height=WINDOW_SIZE[1],
        min_width=MIN_WINDOW_SIZE[0],
        min_height=MIN_WINDOW_SIZE[1],
    )
    dpg.setup_dearpygui()
    dpg.show_viewport()
    # Fill primary monitor: near-full size, then OS maximize for maximum usable area
    dpg.configure_viewport(0, width=WINDOW_SIZE[0], height=WINDOW_SIZE[1])
    try:
        dpg.maximize_viewport()
    except Exception:
        pass

    dpg.set_primary_window("main_window", True)

    # Resize main window to fill viewport when viewport size changes
    def _on_viewport_resize(sender, app_data):
        w, h = dpg.get_viewport_client_width(), dpg.get_viewport_client_height()
        if w > 0 and h > 0:
            dpg.configure_item("main_window", width=w, height=h)

    dpg.set_viewport_resize_callback(_on_viewport_resize)
    # Trigger initial resize so window fills viewport at startup
    _on_viewport_resize(None, None)

    # File dialogs for export
    def _on_png(sender, app_data):
        path = app_data.get("file_path_name")
        if path and _shared.get("last_frame") is not None:
            export_screenshot_png(_shared["last_frame"], Path(path))

    def _on_csv(sender, app_data):
        path = app_data.get("file_path_name")
        if path:
            st = _shared.get("state_front") or _shared.get("state")
            if st is not None:
                export_layer_grids_csv(st, Path(path))

    def _on_npy(sender, app_data):
        # Directory selector: file_path_name or current_path
        path = app_data.get("file_path_name") or app_data.get("current_path")
        if isinstance(path, (list, tuple)) and path:
            path = path[0]
        if path:
            st = _shared.get("state_front") or _shared.get("state")
            if st is not None:
                export_layer_grids_npy(st, Path(path))

    def _on_stim_image(sender, app_data):
        """Load an external image/photo as a stimulus mask."""
        path = app_data.get("file_path_name")
        if isinstance(path, (list, tuple)) and path:
            path = path[0]
        if not path:
            return
        st = _shared.get("state_front") or _shared.get("state")
        if st is None:
            return
        try:
            # Keep RGB so that colors can be binned by L/M/S.
            img = Image.open(path).convert("RGB")
            h, w = st.grid_shape()
            img = img.resize((w, h), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32)
            # Store in 0–1 so spectral construction can preserve RGB ratios.
            st.stimulus_params["image_mask"] = (arr / 255.0).astype(np.float32)
            # Must use stimulus type "image" or the pipeline keeps the previous
            # stimulus (e.g. spot) and ignores image_mask.
            st.stimulus_params["type"] = "image"
            if dpg.does_item_exist("stimulus_type_combo"):
                dpg.set_value("stimulus_type_combo", "image")
            _update_stimulus_visibility("image", st)
        except Exception as e:
            # Fallback: simple stderr print so the app keeps running.
            print(f"Failed to load stimulus image: {e}")

    def _on_rec_save(sender, app_data) -> None:
        path = app_data.get("file_path_name") or app_data.get("current_path")
        if isinstance(path, (list, tuple)) and path:
            path = path[0]
        if not path:
            return
        buf = _shared.get("rec_buffer")
        if buf is None:
            return
        try:
            buf.save(Path(path))
            if dpg.does_item_exist("rec_status_text"):
                dpg.set_value("rec_status_text", f"Saved to {path}")
        except OSError as e:
            print(f"Session save failed: {e}")

    def _on_rec_load(sender, app_data) -> None:
        path = app_data.get("file_path_name") or app_data.get("current_path")
        if isinstance(path, (list, tuple)) and path:
            path = path[0]
        if not path:
            return
        try:
            loaded = load_session_recording(Path(path))
            _shared["loaded_recording"] = loaded
            n = loaded.n_frames
            if dpg.does_item_exist("rec_frame_slider"):
                dpg.configure_item("rec_frame_slider", max_value=max(0, n - 1))
                dpg.set_value("rec_frame_slider", 0)
            if dpg.does_item_exist("rec_status_text"):
                dpg.set_value("rec_status_text", f"Loaded {n} frames")
            _apply_playback_frame()
        except OSError as e:
            print(f"Session load failed: {e}")

    with dpg.file_dialog(
        callback=_on_png,
        tag="file_dialog_png",
        show=False,
        modal=True,
        directory_selector=False,
        height=520,
    ):
        dpg.add_file_extension(".*")
        dpg.add_file_extension(".png", color=(0, 255, 0, 255))

    with dpg.file_dialog(
        callback=_on_csv,
        tag="file_dialog_csv",
        show=False,
        modal=True,
        directory_selector=False,
        height=520,
    ):
        dpg.add_file_extension(".*")
        dpg.add_file_extension(".csv", color=(0, 255, 0, 255))

    with dpg.file_dialog(
        callback=_on_npy,
        tag="file_dialog_npy",
        show=False,
        modal=True,
        directory_selector=True,
        height=520,
    ):
        pass

    with dpg.file_dialog(
        callback=_on_stim_image,
        tag="stim_image_dialog",
        show=False,
        modal=True,
        directory_selector=False,
        height=520,
    ):
        dpg.add_file_extension(".png", color=(0, 255, 0, 255))
        dpg.add_file_extension(".jpg")
        dpg.add_file_extension(".jpeg")
        dpg.add_file_extension(".*")

    with dpg.file_dialog(
        callback=_on_rec_save,
        tag="file_dialog_rec_save",
        show=False,
        modal=True,
        directory_selector=True,
        height=520,
    ):
        pass

    with dpg.file_dialog(
        callback=_on_rec_load,
        tag="file_dialog_rec_load",
        show=False,
        modal=True,
        directory_selector=True,
        height=520,
    ):
        pass

    # Shared state for main loop (double-buffer when worker used; else single state)
    _shared["state"] = state  # legacy alias
    _shared["state_front"] = state
    _shared["state_back"] = state_back
    _shared["sim_on_main_thread"] = SIM_ON_MAIN_THREAD
    _shared["sim_tick_every_n"] = SIM_TICK_EVERY_N
    _shared["sim_tick_counter"] = 0
    _shared["last_frame"] = None
    _shared["connectivity_dirty"] = False
    _shared["frame_count"] = 0  # for deferred resize at startup
    _shared["rgc_fr_history"] = []  # for sparkline (last 100 ticks)
    _shared["stats_tick"] = 0  # throttle stats update
    _shared["all_layers_rgba"] = None  # composite canvas for 2D All Layers view
    _shared["rec_buffer"] = SessionRecordingBuffer()
    _shared["loaded_recording"] = None
    _shared["rec_prev_enabled"] = False
    _shared["playback_active"] = False

    if not SIM_ON_MAIN_THREAD and state_back is not None:
        threading.Thread(target=_sim_worker, daemon=True).start()

    # Main loop: step simulation, render 2D, blit into DPG dynamic texture.
    last_time = dpg.get_total_time()
    while dpg.is_dearpygui_running():
        frame_start = time.perf_counter()
        now = dpg.get_total_time()
        dt = now - last_time
        last_time = now

        # Clamp dt to avoid huge steps after window moves, etc.
        dt = max(1e-3, min(dt, 1 / 30))

        # Keep main window and center panel sized so all three panels fit with no horizontal scroll
        vw = dpg.get_viewport_client_width()
        vh = dpg.get_viewport_client_height()
        if vw > 0 and vh > 0:
            # First 10 frames: if viewport opened small, force default size
            fc = _shared.get("frame_count", 0)
            _shared["frame_count"] = fc + 1
            if fc < 10 and (vw < WINDOW_SIZE[0] or vh < WINDOW_SIZE[1]):
                dpg.configure_viewport(0, width=WINDOW_SIZE[0], height=WINDOW_SIZE[1])
                vw, vh = WINDOW_SIZE[0], WINDOW_SIZE[1]
            dpg.configure_item("main_window", width=vw, height=vh)
            # Pin center width; reserve pixels for borders/spacing so no horizontal scroll
            slack = 32  # borders and gaps between panels
            center_w = max(
                MIN_VIEWPORT_WIDTH,
                vw - LEFT_PANEL_WIDTH - RIGHT_PANEL_WIDTH - slack,
            )
            dpg.configure_item(VIEWPORT_AREA_TAG, width=center_w)

            # Size and center the image in the middle panel (max space, centered)
            if dpg.does_item_exist(VIEWPORT_IMAGE_TAG):
                try:
                    rect_min = dpg.get_item_rect_min(VIEWPORT_AREA_TAG)
                    rect_max = dpg.get_item_rect_max(VIEWPORT_AREA_TAG)
                    area_w = max(1, int(rect_max[0] - rect_min[0]))
                    area_h = max(1, int(rect_max[1] - rect_min[1]))
                except Exception:
                    area_w, area_h = center_w, vh - 60  # fallback
                scale = min(area_w / max(display_w, 1), area_h / max(display_h, 1))
                img_w = int(display_w * scale)
                img_h = int(display_h * scale)
                pos_x = max(0, (area_w - img_w) // 2)
                pos_y = max(0, (area_h - img_h) // 2)
                dpg.configure_item(
                    VIEWPORT_IMAGE_TAG,
                    width=img_w,
                    height=img_h,
                    pos=[pos_x, pos_y],
                )

        playback_active = (
            _shared.get("loaded_recording") is not None
            and dpg.does_item_exist("rec_playback")
            and dpg.get_value("rec_playback")
        )
        _shared["playback_active"] = playback_active

        # Use display buffer (worker writes here) or single state (sim on main thread)
        if not playback_active and _shared.get("sim_on_main_thread"):
            state = _shared["state_front"]
            _shared["state"] = state
            # Tick on main thread every Nth frame to keep 60 FPS
            ctr = _shared.get("sim_tick_counter", 0)
            _shared["sim_tick_counter"] = (ctr + 1) % _shared.get("sim_tick_every_n", 1)
            if ctr == 0:
                tick(state, min(dt, 1.0 / 30.0))
        elif not playback_active:
            state = _shared["state_front"]
            _shared["state"] = state
        else:
            state = _shared["state_front"]
            _shared["state"] = state
            _apply_playback_frame()

        if not playback_active:
            rec_buf = _shared.get("rec_buffer")
            if (
                rec_buf is not None
                and dpg.does_item_exist("rec_enabled")
                and dpg.get_value("rec_enabled")
            ):
                if not _shared.get("rec_prev_enabled", False):
                    rec_buf.clear()
                try:
                    rec_buf.stride = max(1, int(dpg.get_value("rec_stride")))
                except (TypeError, ValueError):
                    rec_buf.stride = 4
                rec_buf.append_frame(state)
        _shared["rec_prev_enabled"] = (
            bool(dpg.get_value("rec_enabled")) if dpg.does_item_exist("rec_enabled") else False
        )

        # Render: 2D view modes share the same dynamic texture.
        view_mode = dpg.get_value("view_mode_combo") if dpg.does_item_exist("view_mode_combo") else "2D Heatmap"
        if view_mode == "2D All Layers":
            # No layer-specific overlays or convergence notes in composite mode.
            if dpg.does_item_exist("layer_convergence_note"):
                dpg.set_value("layer_convergence_note", "")
            comp_rgba = _render_all_layers_composite(state)
            rgba = _resize_rgba_to_display(
                comp_rgba, display_h, display_w, ALL_LAYERS_BG_RGBA
            )
        else:
            # 2D single-layer heatmap: combo value is display label; resolve to internal key.
            layer_display = dpg.get_value("layer_combo") if dpg.does_item_exist("layer_combo") else LAYER_KEY_TO_DISPLAY.get("RGC Firing (L)", "RGC Firing (L)")
            layer_name = LAYER_DISPLAY_TO_KEY.get(layer_display, layer_display)
            if dpg.does_item_exist("layer_convergence_note"):
                if dpg.does_item_exist("show_convergence_ratios") and dpg.get_value("show_convergence_ratios"):
                    _set_convergence_note(layer_name)
                else:
                    dpg.set_value("layer_convergence_note", "")
            if layer_name == "Stimulus":
                rgba = _render_stimulus_rgba(state)
            elif layer_name in ("Cones L", "Cones M", "Cones S"):
                # Inverted: activity = 1 - cone; shared max = 1.0 (same as All Layers).
                cone_L = np.asarray(state.cone_L if state.cone_L is not None else np.zeros(state.grid_shape(), dtype=np.float32), dtype=np.float32).copy()
                cone_M = np.asarray(state.cone_M if state.cone_M is not None else np.zeros(state.grid_shape(), dtype=np.float32), dtype=np.float32).copy()
                cone_S = np.asarray(state.cone_S if state.cone_S is not None else np.zeros(state.grid_shape(), dtype=np.float32), dtype=np.float32).copy()
                if dpg.does_item_exist("biological_scale_2d") and dpg.get_value("biological_scale_2d"):
                    for key, arr in [("Cones L", cone_L), ("Cones M", cone_M), ("Cones S", cone_S)]:
                        dkey = LAYER_KEY_TO_DENSITY.get(key)
                        if dkey and dkey in RELATIVE_DENSITY:
                            scale = RELATIVE_DENSITY["rgc"] / RELATIVE_DENSITY[dkey]
                            arr *= scale
                grid = cone_L if layer_name == "Cones L" else (cone_M if layer_name == "Cones M" else cone_S)
                grid = 1.0 - np.clip(grid, 0.0, 1.0)
                rgba = _grid_to_rgba_absolute_firing(grid, 1.0, _get_heatmap_colormap())
            else:
                rgba = _render_layer_rgba(state, layer_name)
            # Scale bar (100 µm default; Masland 2012, Curcio et al. 1992)
            draw_scale_bar_rgba(
                rgba,
                microns_per_px=state.config.retina.microns_per_px,
                scale_bar_um=float(getattr(state.config.viewer_3d, "scale_bar_um", 100.0)),
                position="bottom_left",
            )
            # Display: block-average downsample if grid > MAX_DISPLAY_SIDE, else upscale
            gh, gw = rgba.shape[0], rgba.shape[1]
            if gh > MAX_DISPLAY_SIDE or gw > MAX_DISPLAY_SIDE:
                rgba = block_average_downsample_rgba(rgba, MAX_DISPLAY_SIDE)
            else:
                rgba = np.repeat(np.repeat(rgba, DISPLAY_SCALE, axis=0), DISPLAY_SCALE, axis=1)

        tex_data = np.ascontiguousarray(rgba.astype(np.float32)).flatten()
        img = (rgba * 255).astype(np.uint8)

        _shared["last_frame"] = img
        dpg.set_value(VIEWPORT_TEX_TAG, tex_data)
        # Update stats every 5th frame to reduce CPU (sparkline, histogram are costly)
        st = _shared.get("stats_tick", 0)
        _shared["stats_tick"] = st + 1
        if st % 5 == 0:
            _update_stats(state)

        dpg.render_dearpygui_frame()

        # Cap frame rate to avoid burning CPU and keep system responsive
        elapsed = time.perf_counter() - frame_start
        sleep_time = (1.0 / TARGET_FPS) - elapsed
        if sleep_time > 0.001:
            time.sleep(sleep_time)

    dpg.destroy_context()

