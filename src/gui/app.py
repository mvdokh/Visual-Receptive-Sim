from __future__ import annotations

"""
Dear PyGui application wiring together:
- Simulation state + pipeline
- 2D viewport (single layer or all-layers mosaic)
- Control / analysis panels
"""

import ctypes
import os
import platform
import random
import threading
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple, cast

# Run simulation on main thread (no background worker) for smoother 60 FPS; set SIM_ON_MAIN_THREAD=1
SIM_ON_MAIN_THREAD = os.environ.get("SIM_ON_MAIN_THREAD", "").strip().lower() in ("1", "true", "yes")
# Tick every Nth frame when on main thread to hit 60 FPS (SIM_TICK_EVERY_N=2 → 30 Hz sim)
SIM_TICK_EVERY_N = max(1, int(os.environ.get("SIM_TICK_EVERY_N", "1")))

import dearpygui.dearpygui as dpg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Upscale factor for display (texture is grid_resolution * DISPLAY_SCALE).
DISPLAY_SCALE = 4
# Right-panel Stats tab + cone/opponent plots: update every N UI frames (1 = each frame).
_STATS_UI_EVERY_N_FRAMES = 1
# Main loop FPS cap.
TARGET_FPS = 60
# PIL: LANCZOS is sharp but slow on megapixel resizes (2D All Layers + letterbox).
_PIL_FAST_RESAMPLE_PIXELS = 350_000
# Dynamics tab: mean-FR trace + spatial FR histogram (wall clock), 10 Hz.
RGC_DYNAMICS_UI_INTERVAL_S = 0.1


def _rgc_grid_mean_fr(arr: np.ndarray) -> float:
    try:
        from hot_numerical.grid_ops import mean_f32_2d

        return float(mean_f32_2d(np.ascontiguousarray(arr, dtype=np.float32)))
    except ImportError:
        return float(np.mean(arr))


def _rgc_fr_hist_plot_data(
    arr: np.ndarray, *, max_hist_points: int = 2048
) -> tuple[list[float], list[float], float, float] | None:
    """
    Bin centers, bar heights, and x-axis limits for the spatial FR histogram.
    Uses Cython subsampled 16-bin path when ``hot_numerical.grid_ops`` is built.
    """
    try:
        from hot_numerical.grid_ops import fr_histogram_16bins_subsampled

        counts, e0, e1 = fr_histogram_16bins_subsampled(
            np.ascontiguousarray(arr, dtype=np.float32), max_hist_points
        )
        n = int(counts.shape[0])
        span = float(e1) - float(e0)
        w = span / float(n) if n else 1.0
        xs = [float(e0) + (i + 0.5) * w for i in range(n)]
        ys = [float(counts[i]) for i in range(n)]
        return xs, ys, float(e0), float(e1)
    except ImportError:
        pass
    flat = arr.flatten()
    flat = flat[np.isfinite(flat)]
    if len(flat) > max_hist_points:
        step = len(flat) // max_hist_points
        flat = flat[::step][:max_hist_points]
    if len(flat) == 0:
        return None
    mn, mx = float(np.min(flat)), float(np.max(flat))
    if not np.isfinite(mn):
        mn = 0.0
    if not np.isfinite(mx):
        mx = mn + 1.0
    if mx <= mn:
        mx = mn + 1.0
        bins = 2
    else:
        min_range = max(1e-9, np.finfo(np.float64).tiny * 20)
        if (mx - mn) < min_range:
            mx = mn + 1.0
            bins = 2
        else:
            bins = 16
    try:
        counts, edges = np.histogram(flat, bins=bins, range=(mn, mx))
    except ValueError:
        bins = 2
        mx = mn + 1.0
        counts, edges = np.histogram(flat, bins=bins, range=(mn, mx))
    xs = [(float(edges[i]) + float(edges[i + 1])) / 2 for i in range(bins)]
    ys = [float(c) for c in counts]
    return xs, ys, float(edges[0]), float(edges[-1])


from src.config import SpatialHeterogeneityMode, default_config
from src.rendering.heatmap import (
    block_average_downsample_rgba,
    composite_spatial_heterogeneity_overlays,
    draw_scale_bar_rgba,
    grid_to_rgba,
    spectrum_to_stimulus_rgba,
)
from src.simulation.stimulus.spectral import build_stimulus_spectrum
from src.gui.app_context import AppContext
from src.gui.layout_constants import (
    ALL_LAYERS_VIEWER_MARGIN_PX,
    CONN_WEIGHT_MAX,
    CONN_WEIGHT_MIN,
    LEFT_STACK_SCROLL_TAG,
    LEFT_PANEL_FONT_PX,
    LEFT_PANEL_WIDTH,
    MIN_VIEWPORT_WIDTH,
    MIN_WINDOW_SIZE,
    RIGHT_PANEL_WIDTH,
    SIDE_COMBO_WIDTH,
    SIDE_PANEL_TEXT_WRAP,
    SIDE_SLIDER_WIDTH,
    SIDE_SPIN_WIDE,
    SIDE_SPIN_WIDTH,
    STIM_COMBO_WIDTH,
    STIM_SLIDER_WIDTH,
    VIEWPORT_AREA_TAG,
    VIEWPORT_AREA_THEME_TAG,
    VIEWPORT_BG_RGBA,
    VIEWPORT_IMAGE_TAG,
    VIEWPORT_PANEL_BG_RGB_U8,
    VIEWPORT_TEX_TAG,
)
from src.gui import layout as gui_layout
from src.gui import settings as user_settings
from src.gui import themes as gui_themes
from src.gui.panels.data_export import (
    export_layer_grids_csv,
    export_layer_grids_npy,
    export_screenshot_png,
)
from src.gui.panels.stats_plots import STAT_LAYER_PLOT_ORDER
from src.simulation import SimState, tick
from src.simulation.spatial_heterogeneity_maps import rebuild_spatial_heterogeneity
from src.simulation.session_recording import (
    LoadedSessionRecording,
    SessionRecordingBuffer,
    load_session_recording,
)
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
    """Set convergence overlay text (ASCII only: bundled UI font may not draw Unicode arrows)."""
    if not dpg.does_item_exist("layer_convergence_note"):
        return
    notes = {
        "Stimulus": "Stimulus to photoreceptors",
        "Cones L": f"Photoreceptors to bipolar: ~{int(PHOTORECEPTOR_RGC_RATIO)}:1 overall convergence",
        "Cones M": f"Rod:cone ~{int(ROD_CONE_RATIO)}:1  |  Cones to bipolar ~{int(PHOTORECEPTOR_RGC_RATIO)}:1",
        "Cones S": f"Rods ~{RODS_TOTAL//1_000_000}M : Cones ~{CONES_TOTAL//1_000_000}M  |  20:1",
        "Horizontal": f"Horizontal ~{int(INL_FRAC_HORIZONTAL*100)}% INL  |  sparse lateral",
        "Bipolar ON": f"Bipolar ~{int(INL_FRAC_BIPOLAR*100)}% INL  |  Bipolar to RGC ~100:1",
        "Amacrine": f"Amacrine ~{int(INL_FRAC_AMACRINE*100)}% INL  |  Bipolar to RGC ~100:1",
        "RGC Firing (L)": f"~{RGCS_TOTAL//1_000_000}M RGCs  |  ~{int(PHOTORECEPTOR_RGC_RATIO)}:1 photoreceptor to RGC",
    }
    dpg.set_value("layer_convergence_note", notes.get(layer_name, ""))


def _primary_screen_size_px() -> Optional[Tuple[int, int]]:
    """
    Best-effort physical/logical pixel size of the primary display.

    Avoids tkinter/Tcl: creating and destroying a Tk root at import time can leave
    the Tcl interpreter in a bad state on macOS and crash the process (e.g. Dock
    focus) while Dear PyGui is running.
    """
    env = os.environ.get("VISUAL_RECEPTIVE_SIM_SCREEN", "").strip().lower()
    if env and "x" in env:
        try:
            a, b = env.split("x", 1)
            w, h = int(a.strip()), int(b.strip())
            if w > 0 and h > 0:
                return w, h
        except ValueError:
            pass

    sysname = platform.system()
    if sysname == "Darwin":
        try:
            cg = ctypes.CDLL(
                "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
            )
            CGMainDisplayID = cg.CGMainDisplayID
            CGMainDisplayID.restype = ctypes.c_uint32
            CGMainDisplayID.argtypes = []

            class _CGPoint(ctypes.Structure):
                _fields_ = (("x", ctypes.c_double), ("y", ctypes.c_double))

            class _CGSize(ctypes.Structure):
                _fields_ = (("width", ctypes.c_double), ("height", ctypes.c_double))

            class _CGRect(ctypes.Structure):
                _fields_ = (("origin", _CGPoint), ("size", _CGSize))

            CGDisplayBounds = cg.CGDisplayBounds
            CGDisplayBounds.argtypes = (ctypes.c_uint32,)
            CGDisplayBounds.restype = _CGRect
            rect = CGDisplayBounds(CGMainDisplayID())
            w = int(rect.size.width)
            h = int(rect.size.height)
            if w > 0 and h > 0:
                return w, h
        except Exception:
            return None
    if sysname == "Windows":
        try:
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            w = int(user32.GetSystemMetrics(0))
            h = int(user32.GetSystemMetrics(1))
            if w > 0 and h > 0:
                return w, h
        except Exception:
            return None
    return None


def _default_window_size() -> Tuple[int, int]:
    """Primary monitor size minus margin for title bar / dock; fallback for headless."""
    raw = _primary_screen_size_px()
    pad_w, pad_h = 16, 72
    if raw is not None:
        w, h = raw
        return (
            max(MIN_WINDOW_SIZE[0], w - pad_w),
            max(MIN_WINDOW_SIZE[1], h - pad_h),
        )
    return (max(MIN_WINDOW_SIZE[0], 1680), max(MIN_WINDOW_SIZE[1], 1000))


# Default size before maximize; also used as minimum target in the first frames
WINDOW_SIZE: Tuple[int, int] = _default_window_size()
DATA_EXPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "exports"

# Shared state for export callbacks, RF compute, and mouse orbit (updated each frame)
_shared: dict = {}

# Max display resolution for 2D viewer (block-average downsampling above this)
MAX_DISPLAY_SIDE = 1024
# 2D All Layers composite is letterboxed into the viewport texture; allow a larger
# bitmap than single-layer mode so wheel-zoom does not upscale a heavily downsampled image.
MAX_COMPOSITE_TEXTURE_CAP = 2048

try:
    _PIL_RESAMPLE_HIGH = Image.Resampling.LANCZOS
    _PIL_RESAMPLE_FAST = Image.Resampling.BILINEAR
except AttributeError:
    _PIL_RESAMPLE_HIGH = getattr(Image, "LANCZOS", Image.BILINEAR)
    _PIL_RESAMPLE_FAST = Image.BILINEAR


def _pil_resample(in_h: int, in_w: int, out_h: int, out_w: int):
    """LANCZOS for small images; bilinear when resizing large bitmaps (viewport / All Layers)."""
    if in_h * in_w > _PIL_FAST_RESAMPLE_PIXELS or out_h * out_w > _PIL_FAST_RESAMPLE_PIXELS:
        return _PIL_RESAMPLE_FAST
    return _PIL_RESAMPLE_HIGH

# RGC spike raster (right panel): rolling window in sim time; one neuron per row (ticks).
RASTER_TIME_WINDOW_S = 4.0
RASTER_MAX_EVENTS = 200_000
# Max subsampled neurons (~sqrt rows); stride chosen so grid fits under this cap.
RASTER_MAX_NEURONS = 576
# Half-height of each spike tick in neuron-index units (vertical line in plot space).
RASTER_TICK_HALF_HEIGHT = 0.42

# Center viewer: wheel zoom vs fit-to-panel scale (global handler_registry + viewport hover check).
VIEWPORT_ZOOM_MIN = 0.25
VIEWPORT_ZOOM_MAX = 12.0
VIEWPORT_ZOOM_STEP = 1.12
VIEWPORT_ZOOM_WHEEL_REG = "viewport_zoom_wheel_reg"


def _composite_layout_dims(grid_h: int, grid_w: int) -> tuple[int, int, int, int]:
    """Native composite size: comp_h, comp_w, label_strip_h, row_gap."""
    label_strip_h = max(12, min(18, int(grid_h * 0.055)))
    row_gap = 2
    comp_w = 3 * grid_w
    comp_h = 4 * grid_h + 4 * label_strip_h + 3 * row_gap
    return comp_h, comp_w, label_strip_h, row_gap


def _base_panel_widths_from_preset() -> Tuple[int, int]:
    comp = _shared.get("layout_composer")
    if comp is not None:
        lw = int(comp.preset.get("left_width", LEFT_PANEL_WIDTH))
        rw = int(comp.preset.get("right_width", RIGHT_PANEL_WIDTH))
        return lw, rw
    return LEFT_PANEL_WIDTH, RIGHT_PANEL_WIDTH


def _apply_side_panel_widths_to_ui(lw: int, rw: int) -> None:
    """Resize left scroll + panel child_windows and the right stats column."""
    plw = max(220, int(lw))
    prw = max(220, int(rw))
    if dpg.does_item_exist(LEFT_STACK_SCROLL_TAG):
        dpg.configure_item(LEFT_STACK_SCROLL_TAG, width=plw)
    for tag in ("panel_circuit_root", "panel_heterogeneity_root"):
        if dpg.does_item_exist(tag):
            dpg.configure_item(tag, width=plw)
    if dpg.does_item_exist("panel_stats_plots_root"):
        dpg.configure_item("panel_stats_plots_root", width=prw)


def _texture_target_hw(state: SimState) -> tuple[int, int]:
    """Dynamic texture (W, H) large enough for single-layer and All Layers letterboxing."""
    gh, gw = state.grid_shape()
    if gh > MAX_DISPLAY_SIDE or gw > MAX_DISPLAY_SIDE:
        sw, sh = MAX_DISPLAY_SIDE, MAX_DISPLAY_SIDE
    else:
        sw = min(MAX_DISPLAY_SIDE, gw * DISPLAY_SCALE)
        sh = min(MAX_DISPLAY_SIDE, gh * DISPLAY_SCALE)
    ch, cw, _, _ = _composite_layout_dims(gh, gw)
    cap = MAX_COMPOSITE_TEXTURE_CAP
    scale = min(cap / max(cw, 1), cap / max(ch, 1))
    al_w = max(1, int(round(cw * scale)))
    al_h = max(1, int(round(ch * scale)))
    return max(sw, al_w), max(sh, al_h)


def _ui_letterbox_rgba() -> Tuple[float, float, float, float]:
    v = _shared.get("ui_child_bg_rgba", VIEWPORT_BG_RGBA)
    if isinstance(v, (list, tuple)) and len(v) >= 4:
        return (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
    return cast(Tuple[float, float, float, float], tuple(VIEWPORT_BG_RGBA))


def _refresh_viewport_chrome_theme_from_tokens(tokens: gui_themes.ThemeTokens) -> None:
    """Center viewer child_window fill matches theme ChildBg (side panels).

    The theme is created once in ``viewport.build`` with a fixed tag; Dear PyGui
    rejects recreating that tag (alias still registered). Update existing
    ``mvThemeColor`` children in place instead of delete/recreate.
    """
    if not dpg.does_item_exist(VIEWPORT_AREA_TAG) or not dpg.does_item_exist(VIEWPORT_AREA_THEME_TAG):
        return
    rgb = tuple(int(c) for c in tokens.child_bg[:4])
    rgb_list = list(rgb)
    try:
        for si in range(4):
            comps = dpg.get_item_children(VIEWPORT_AREA_THEME_TAG, si)
            if not comps:
                continue
            for comp in comps:
                for sj in range(4):
                    leaves = dpg.get_item_children(comp, sj)
                    if not leaves:
                        continue
                    for leaf in leaves:
                        try:
                            dpg.configure_item(leaf, default_value=rgb_list)
                        except Exception:
                            try:
                                dpg.configure_item(leaf, value=rgb_list)
                            except Exception:
                                try:
                                    dpg.set_value(leaf, rgb_list)
                                except Exception:
                                    pass
    except Exception:
        return
    try:
        dpg.bind_item_theme(VIEWPORT_AREA_TAG, VIEWPORT_AREA_THEME_TAG)
    except Exception:
        pass


def _mouse_wheel_vertical_delta(app_data) -> float:
    """Normalize Dear PyGui / ImGui wheel payload to a signed step count (≈1 per notch)."""
    if app_data is None:
        return 0.0
    x = app_data
    if isinstance(x, (list, tuple)):
        x = x[1] if len(x) > 1 else x[0]
    elif isinstance(x, dict):
        x = x.get("MouseWheel", x.get("vertical", x.get("Vertical", 0)))
    try:
        raw = float(x)
    except (TypeError, ValueError):
        return 0.0
    if abs(raw) > 10:
        raw /= 120.0
    return max(-12.0, min(12.0, raw))


def _on_viewport_mouse_wheel(sender, app_data) -> None:
    # Mouse wheel handlers must live under handler_registry (not item_handler_registry).
    # Registry is global, so only react when the cursor is over the center viewer.
    if not dpg.does_item_exist(VIEWPORT_AREA_TAG):
        return
    try:
        over = dpg.is_item_hovered(VIEWPORT_AREA_TAG)
        if not over and dpg.does_item_exist(VIEWPORT_IMAGE_TAG):
            over = dpg.is_item_hovered(VIEWPORT_IMAGE_TAG)
        if not over:
            return
    except Exception:
        return
    dz = _mouse_wheel_vertical_delta(app_data)
    if abs(dz) < 1e-6:
        return
    z = float(_shared.get("viewport_zoom", 1.0))
    z *= VIEWPORT_ZOOM_STEP**dz
    _shared["viewport_zoom"] = max(VIEWPORT_ZOOM_MIN, min(VIEWPORT_ZOOM_MAX, z))


def _viewport_hovered() -> bool:
    if not dpg.does_item_exist(VIEWPORT_AREA_TAG):
        return False
    try:
        if dpg.is_item_hovered(VIEWPORT_AREA_TAG):
            return True
        if dpg.does_item_exist(VIEWPORT_IMAGE_TAG) and dpg.is_item_hovered(VIEWPORT_IMAGE_TAG):
            return True
    except Exception:
        pass
    return False


def _shift_held() -> bool:
    return bool(dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift))


def _viewport_shift_pan_tick() -> None:
    """Shift + left-drag: pan the viewer (offset stored in ``viewport_pan_px``)."""
    if not _shift_held() or not dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
        _shared["viewport_pan_dragging"] = False
        _shared["_vp_pan_prev_mouse"] = None
        return
    if not _shared.get("viewport_pan_dragging"):
        if not _viewport_hovered():
            return
        _shared["viewport_pan_dragging"] = True
        mx, my = dpg.get_mouse_pos(local=False)
        _shared["_vp_pan_prev_mouse"] = [float(mx), float(my)]
        return
    mx, my = dpg.get_mouse_pos(local=False)
    prev = _shared.get("_vp_pan_prev_mouse")
    if prev is None:
        _shared["_vp_pan_prev_mouse"] = [float(mx), float(my)]
        return
    dx = float(mx) - float(prev[0])
    dy = float(my) - float(prev[1])
    _shared["_vp_pan_prev_mouse"] = [float(mx), float(my)]
    pan = _shared.get("viewport_pan_px", [0.0, 0.0])
    _shared["viewport_pan_px"] = [float(pan[0]) + dx, float(pan[1]) + dy]


def _reset_main_viewer_view() -> None:
    _shared.update(
        {
            "viewport_zoom": 1.0,
            "viewport_pan_px": [0.0, 0.0],
            "viewport_pan_dragging": False,
            "_vp_pan_prev_mouse": None,
        }
    )


def _toggle_spike_raster_ui(enabled: bool) -> None:
    """Show or hide the raster plot and clear buffered points when spikes are disabled."""
    show = bool(enabled)
    if dpg.does_item_exist("spike_raster_plot_group"):
        dpg.configure_item("spike_raster_plot_group", show=show)
    if not show:
        dq = _shared.get("raster_events")
        if isinstance(dq, deque):
            dq.clear()
        if dpg.does_item_exist("series_rgc_raster"):
            dpg.set_value("series_rgc_raster", [[], []])


def _raster_spike_attr_for_heatmap() -> str:
    """Spike grid that matches the 'RGC spikes (L)' 2D heatmap layer."""
    if not dpg.does_item_exist("layer_combo"):
        return "spike_midget_on_L"
    label = str(dpg.get_value("layer_combo"))
    layer_key = LAYER_DISPLAY_TO_KEY.get(label)
    if layer_key == "RGC spikes (L)":
        return "spike_midget_on_L"
    return "spike_midget_on_L"


def _raster_subsample_stride(n: int) -> tuple[int, int, int]:
    """Stride and subsampled (h, w) so h*w <= RASTER_MAX_NEURONS (row-major matches heatmap)."""
    max_side = max(8, int(np.sqrt(RASTER_MAX_NEURONS)))
    stride = max(1, (n + max_side - 1) // max_side)
    sub_h = (n + stride - 1) // stride
    sub_w = (n + stride - 1) // stride
    return stride, sub_h, sub_w


def _update_spike_raster_series(state: SimState) -> None:
    """
    Append subsampled spike events for the same map as the RGC spikes heatmap.

    Each neuron is one horizontal row (y); spikes draw as short vertical ticks in time.
    Rows use row-major indices with y flipped so row 0 matches the top of the heatmap.
    Spikes in the same sim step are spread across [t-dt, t] so they do not collapse to one x.
    """
    if not state.config.spike_output.enabled:
        return
    if not dpg.does_item_exist("series_rgc_raster"):
        return
    n = int(state.config.retina.grid_resolution)
    attr = _raster_spike_attr_for_heatmap()
    arr = getattr(state, attr, None)
    if arr is None or arr.shape != (n, n):
        return
    stride, sub_h, sub_w = _raster_subsample_stride(n)
    k = sub_h * sub_w
    sub = np.asarray(arr[::stride, ::stride], dtype=np.float32).ravel()
    idx = np.flatnonzero(sub > 0.5)
    t = float(state.time)
    t_min = t - RASTER_TIME_WINDOW_S
    dq = _shared.get("raster_events")
    if not isinstance(dq, deque):
        dq = deque()
        _shared["raster_events"] = dq

    if idx.size > 0:
        idx.sort()
        sim_dt = float(_shared.get("last_sim_dt", 1.0 / 60.0))
        t0 = t - sim_dt
        n_sp = int(idx.size)
        for j, lin in enumerate(idx):
            t_evt = t0 + (float(j) + 0.5) / float(max(n_sp, 1)) * sim_dt
            y_row = float((k - 1) - int(lin))
            dq.append((t_evt, y_row))
    while dq and cast(Tuple[float, float], dq[0])[0] < t_min:
        dq.popleft()
    while len(dq) > RASTER_MAX_EVENTS:
        dq.popleft()

    if dpg.does_item_exist("plot_raster_ax"):
        dpg.set_axis_limits("plot_raster_ax", max(0.0, t - RASTER_TIME_WINDOW_S), max(RASTER_TIME_WINDOW_S, t))
    if dpg.does_item_exist("plot_raster_ay"):
        dpg.set_axis_limits("plot_raster_ay", -0.5, max(0.5, float(k) - 0.5))

    if not dq:
        dpg.set_value("series_rgc_raster", [[], []])
        return
    nan = float("nan")
    h = float(RASTER_TICK_HALF_HEIGHT)
    xs_list: List[float] = []
    ys_list: List[float] = []
    for t_evt, y_row in dq:
        xs_list.extend([t_evt, t_evt, nan])
        ys_list.extend([y_row - h, y_row + h, nan])
    dpg.set_value("series_rgc_raster", [xs_list, ys_list])


def _render_stimulus_rgba(state: SimState) -> np.ndarray:
    """Return Stimulus layer as (H, W, 4) float32 RGBA in 0–1.

    Always derives the pattern from ``stimulus_params`` (via ``build_stimulus_spectrum``)
    so the viewport matches the UI even when ``tick`` is skipped (e.g. ``SIM_TICK_EVERY_N`` > 1).
    ``state.stimulus_spectrum`` can lag one or more frames behind user edits.
    """
    stim_type = state.stimulus_params.get("type", "spot")
    h, w = state.grid_shape()
    wl = state.config.spectral.wavelengths
    # For image stimuli, show the loaded RGB image directly so the
    # user sees the true pixel colors rather than the spectral
    # centroid approximation used internally for cones.
    if stim_type == "image" and "image_mask" in state.stimulus_params:
        from skimage.transform import resize as sk_resize

        img = np.asarray(state.stimulus_params["image_mask"], dtype=np.float32)
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
        gain = float(state.stimulus_params.get("intensity", 1.0))
        rgb = np.clip(img * gain, 0.0, 1.0)
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        rgba[..., :3] = rgb
        rgba[..., 3] = 1.0
        return rgba

    spec = build_stimulus_spectrum(
        state.stimulus_params,
        state.config.spectral,
        (h, w),
        time_s=float(state.time),
        retina=state.config.retina,
    )
    return spectrum_to_stimulus_rgba(spec, wl)


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
    img_pil = img_pil.resize((w, h), _pil_resample(rgba.shape[0], rgba.shape[1], h, w))
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
    ui = _ui_letterbox_rgba()
    strip = tuple(int(ui[i] * 255.0) for i in range(3)) + (255,)
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
    comp_h, comp_w, label_strip_h, row_gap = _composite_layout_dims(grid_h, grid_w)

    bg = np.asarray(_ui_letterbox_rgba(), dtype=np.float32)
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
        img_pil = img_pil.resize(
            (target_w, target_h),
            _pil_resample(h, w, target_h, target_w),
        )
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


def sync_stimulus_type_in_params(state: SimState, new_type: str) -> None:
    """Set ``stimulus_params['type']`` and remove ``image_mask`` when leaving image mode."""
    t = str(new_type)
    state.stimulus_params["type"] = t
    if t != "image":
        state.stimulus_params.pop("image_mask", None)


def apply_stimulus_type_change(state: SimState, new_type: str) -> None:
    """Apply a stimulus type from the UI: params + visibility (Dear PyGui)."""
    sync_stimulus_type_in_params(state, new_type)
    _update_stimulus_visibility(str(new_type), state)


def apply_stimulus_type_change_from_ui(new_type: str) -> None:
    """Combo callback: always mutate ``state_front`` so worker/front-buffer stays in sync."""
    st = _shared.get("state_front") or _shared.get("state")
    if st is None:
        return
    apply_stimulus_type_change(st, new_type)


def _update_view_mode_ui(mode: str) -> None:
    """Show/hide controls depending on active 2D mode."""
    # Layer combo is only meaningful in single-layer 2D heatmap mode.
    if dpg.does_item_exist("layer_combo"):
        if mode == "2D Heatmap":
            dpg.show_item("layer_combo")
        else:
            dpg.hide_item("layer_combo")


def _apply_layout_preset(name: str) -> None:
    comp = _shared.get("layout_composer")
    if comp is None:
        return
    try:
        data = gui_layout.load_preset_dict(gui_layout.preset_path(name))
    except Exception as exc:
        print(f"Layout preset load failed ({name}): {exc}")
        return
    comp.apply_preset(data)
    _shared["active_preset"] = str(name)
    try:
        dpg.configure_viewport(0, min_width=comp.min_viewport_client_width())
    except Exception:
        pass


def _bind_global_ui_theme(theme_id: str) -> None:
    path = gui_themes.theme_json_path(theme_id)
    if not path.is_file():
        path = gui_themes.theme_json_path("dark_plus")
    tokens = gui_themes.load_theme(path)
    _shared["active_theme_id"] = path.stem
    new_tid = gui_themes.build_dpg_theme(tokens)
    dpg.bind_theme(new_tid)
    old = _shared.pop("global_ui_theme_tag", None)
    if old is not None and str(old) != str(new_tid) and dpg.does_item_exist(old):
        try:
            dpg.delete_item(old)
        except Exception:
            pass
    _shared["global_ui_theme_tag"] = new_tid
    _shared["ui_child_bg_rgba"] = tuple(
        float(tokens.child_bg[i]) / 255.0 for i in range(4)
    )
    _refresh_viewport_chrome_theme_from_tokens(tokens)


def _apply_ui_theme_from_menu(theme_id: str) -> None:
    if not gui_themes.theme_json_path(theme_id).is_file():
        print(f"Unknown theme: {theme_id}")
        return
    _bind_global_ui_theme(theme_id)
    _shared["active_theme_id"] = theme_id


def _build_menu_bar() -> None:
    with dpg.menu_bar():
        with dpg.menu(label="File"):
            dpg.add_menu_item(label="Quit", callback=lambda: dpg.stop_dearpygui())
        with dpg.menu(label="Simulation"):
            dpg.add_menu_item(label="Pause / Resume")  # placeholder
        with dpg.menu(label="Settings"):
            dpg.add_combo(
                label="Layout preset",
                items=["default", "wide_viewport", "plots_bottom"],
                default_value="default",
                tag="layout_preset_combo",
                width=220,
                callback=lambda s, a: _apply_layout_preset(str(a)),
            )
            dpg.add_combo(
                label="Color theme",
                items=["dark_plus", "light", "high_contrast", "paper"],
                default_value="dark_plus",
                tag="color_theme_combo",
                width=220,
                callback=lambda s, a: _apply_ui_theme_from_menu(str(a)),
            )
        with dpg.menu(label="Help"):
            dpg.add_menu_item(
                label="Reset main viewer zoom & pan",
                callback=_reset_main_viewer_view,
            )
            dpg.add_menu_item(label="About", callback=lambda: dpg.show_item("about_window"))


def _set_conn_weight(state: SimState, key: str, value: float) -> None:
    if key is None or not isinstance(key, str):
        return
    if hasattr(state.config, "connectivity_weights"):
        v = max(CONN_WEIGHT_MIN, min(CONN_WEIGHT_MAX, float(value)))
        setattr(state.config.connectivity_weights, key, v)


def _set_connectivity_dirty() -> None:
    _shared["connectivity_dirty"] = True


_SH_MODE_LABELS = (
    "Homogeneous",
    "Parameter scatter",
    "Discrete type map",
    "Eccentricity gradient",
    "Voronoi mosaic",
)

_SH_TYPE_BUCKET_LABELS = (
    "Midget ON",
    "Midget OFF",
    "Parasol ON",
    "Parasol OFF",
    "Bistratified",
    "Other",
)


def _sh_mode_index(mode: SpatialHeterogeneityMode) -> int:
    order = (
        SpatialHeterogeneityMode.HOMOGENEOUS,
        SpatialHeterogeneityMode.SCATTER,
        SpatialHeterogeneityMode.TYPE_MAP,
        SpatialHeterogeneityMode.ECCENTRICITY,
        SpatialHeterogeneityMode.MOSAIC,
    )
    return order.index(mode)


def _sh_mark_heterogeneity_dirty(state: SimState) -> None:
    state.heterogeneity_dirty = True
    _shared["connectivity_dirty"] = True


def _sh_sync_mode_group_visibility(active: int) -> None:
    for i in range(len(_SH_MODE_LABELS)):
        tag = f"sh_mode_group_{i}"
        if dpg.does_item_exist(tag):
            dpg.configure_item(tag, show=(i == active))


def _sh_normalize_tm_fractions(state: SimState) -> None:
    tm = state.config.spatial_heterogeneity.type_map
    vals = []
    for i in range(6):
        tag = f"sh_tm_frac_{i}"
        vals.append(float(dpg.get_value(tag)) if dpg.does_item_exist(tag) else 0.0)
    a = np.maximum(np.array(vals, dtype=np.float64), 0.0)
    s = float(a.sum())
    if s <= 1e-12:
        a[:] = 1.0 / 6.0
    else:
        a /= s
    tm.type_fractions = tuple(float(x) for x in a)
    for i in range(6):
        if dpg.does_item_exist(f"sh_tm_frac_{i}"):
            dpg.set_value(f"sh_tm_frac_{i}", float(a[i]))
    if dpg.does_item_exist("sh_tm_frac_readout"):
        parts = [f"{_SH_TYPE_BUCKET_LABELS[i]}: {100.0 * float(a[i]):.1f}%" for i in range(6)]
        dpg.set_value("sh_tm_frac_readout", "  |  ".join(parts))


def _tm_set_rf(state: SimState, i: int, v: float) -> None:
    t = list(state.config.spatial_heterogeneity.type_map.rf_multiplier)
    t[i] = max(0.1, min(4.0, float(v)))
    state.config.spatial_heterogeneity.type_map.rf_multiplier = tuple(t)
    if dpg.does_item_exist(f"sh_tm_rf_{i}"):
        dpg.set_value(f"sh_tm_rf_{i}", t[i])


def _tm_set_gain(state: SimState, i: int, v: float) -> None:
    t = list(state.config.spatial_heterogeneity.type_map.gain_multiplier)
    t[i] = max(0.1, min(4.0, float(v)))
    state.config.spatial_heterogeneity.type_map.gain_multiplier = tuple(t)
    if dpg.does_item_exist(f"sh_tm_gn_{i}"):
        dpg.set_value(f"sh_tm_gn_{i}", t[i])


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


def _randomize_connectivity_weights(state: SimState) -> None:
    cw = state.config.connectivity_weights
    for key in ("cone_to_horizontal", "cone_to_bipolar", "horizontal_to_cone",
                "bipolar_to_amacrine", "amacrine_to_bipolar", "bipolar_to_rgc"):
        setattr(
            cw,
            key,
            random.uniform(CONN_WEIGHT_MIN, CONN_WEIGHT_MAX),
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


def _sync_right_panel_plot_axes() -> None:
    """Apply linear / log / symlog scales from right-panel checkboxes (after limits are set)."""
    lin = dpg.mvPlotScale_Linear
    log10 = dpg.mvPlotScale_Log10
    symlog = getattr(dpg, "mvPlotScale_SymLog", lin)

    def _cfg(tag: str, scale: int) -> None:
        if dpg.does_item_exist(tag):
            try:
                dpg.configure_item(tag, scale=scale)
            except Exception:
                pass

    spark_log_y = dpg.does_item_exist("plot_log_spark_y") and bool(dpg.get_value("plot_log_spark_y"))
    hist_log_x = dpg.does_item_exist("plot_log_hist_x") and bool(dpg.get_value("plot_log_hist_x"))
    hist_log_y = dpg.does_item_exist("plot_log_hist_y") and bool(dpg.get_value("plot_log_hist_y"))
    cone_log_y = dpg.does_item_exist("plot_log_cone_y") and bool(dpg.get_value("plot_log_cone_y"))
    oppo_symlog_y = dpg.does_item_exist("plot_log_oppo_y") and bool(dpg.get_value("plot_log_oppo_y"))

    _cfg("spark_x", lin)
    _cfg("spark_y", log10 if spark_log_y else lin)
    _cfg("hist_x", log10 if hist_log_x else lin)
    _cfg("hist_y", log10 if hist_log_y else lin)
    _cfg("plot_cone_ax", lin)
    _cfg("plot_cone_ay", log10 if cone_log_y else lin)
    _cfg("plot_oppo_ax", lin)
    _cfg("plot_oppo_ay", symlog if oppo_symlog_y else lin)


def _update_rgc_dynamics_plots(state: SimState) -> None:
    """
    RGC mean-FR sparkline and spatial FR histogram: throttled by
    ``RGC_DYNAMICS_UI_INTERVAL_S`` (wall clock) so plots do not refresh every frame.
    """
    if state.fr_midget_on_L is None:
        return
    now = time.perf_counter()
    last = float(_shared.get("rgc_dynamics_last_wall_t", 0.0))
    if last > 0.0 and (now - last) < RGC_DYNAMICS_UI_INTERVAL_S:
        return
    _shared["rgc_dynamics_last_wall_t"] = now

    mean_fr = _rgc_grid_mean_fr(state.fr_midget_on_L)
    hist = _shared.get("rgc_fr_history", [])
    hist.append(mean_fr)
    hist = hist[-100:]
    _shared["rgc_fr_history"] = hist

    if hist and dpg.does_item_exist("sparkline_series"):
        dt = RGC_DYNAMICS_UI_INTERVAL_S
        xs = [i * dt for i in range(len(hist))]
        dpg.set_value("sparkline_series", [xs, hist])
        if dpg.does_item_exist("spark_x") and dpg.does_item_exist("spark_y"):
            mn, mx = min(hist), max(hist)
            pad = max((mx - mn) * 0.1, 1.0) if mx > mn else 1.0
            t_hi = max(dt, (len(hist) - 1) * dt)
            dpg.set_axis_limits("spark_x", 0.0, t_hi)
            spark_log_y = dpg.does_item_exist("plot_log_spark_y") and bool(dpg.get_value("plot_log_spark_y"))
            if spark_log_y:
                if mx <= 0.0:
                    dpg.set_axis_limits("spark_y", 1e-3, 1.0)
                else:
                    ymin = max(1e-6, mn) if mn > 0 else max(1e-6, mx * 1e-4)
                    ymax = max(mx + pad, ymin * 1.01)
                    dpg.set_axis_limits("spark_y", ymin, ymax)
            else:
                dpg.set_axis_limits("spark_y", max(0, mn - pad), mx + pad)

    if dpg.does_item_exist("hist_series"):
        hd = _rgc_fr_hist_plot_data(state.fr_midget_on_L, max_hist_points=2048)
        if hd is not None:
            xs, counts_list, hx_lo, hx_hi = hd
            dpg.set_value("hist_series", [xs, counts_list])
            if dpg.does_item_exist("hist_x") and dpg.does_item_exist("hist_y"):
                hist_log_x = dpg.does_item_exist("plot_log_hist_x") and bool(dpg.get_value("plot_log_hist_x"))
                if hist_log_x and hx_lo <= 0.0:
                    span = max(hx_hi - hx_lo, 1e-9)
                    hx_lo = max(1e-9, hx_hi - span * 0.999)
                dpg.set_axis_limits("hist_x", hx_lo, hx_hi)
                cmax = max(counts_list) if counts_list else 0.0
                hist_log_y = dpg.does_item_exist("plot_log_hist_y") and bool(dpg.get_value("plot_log_hist_y"))
                if hist_log_y:
                    ymin = 0.5 if cmax > 0.0 else 0.1
                    ymax = max(cmax * 1.15, ymin * 10.0, 1.0)
                    dpg.set_axis_limits("hist_y", ymin, ymax)
                else:
                    dpg.set_axis_limits("hist_y", 0.0, max(1.0, cmax * 1.1))

    _sync_right_panel_plot_axes()


def _update_stats(state: SimState) -> None:
    if state.lm_opponent is not None:
        dpg.set_value("lm_summary", f"L-M: mean {float(np.mean(state.lm_opponent)):+.3f}")
    if state.by_opponent is not None:
        dpg.set_value("by_summary", f"S - (L+M): mean {float(np.mean(state.by_opponent)):+.3f}")
    # Plots tab: cone means + opponent trajectories
    if state.cone_L is not None and dpg.does_item_exist("series_cone_bars"):
        mL = float(np.mean(state.cone_L))
        mM = float(np.mean(state.cone_M))
        mS = float(np.mean(state.cone_S))
        dpg.set_value("series_cone_bars", [[0, 1, 2], [mL, mM, mS]])
        ymax = max(1e-9, mL, mM, mS) * 1.15
        if dpg.does_item_exist("plot_cone_ay"):
            cone_log_y = dpg.does_item_exist("plot_log_cone_y") and bool(dpg.get_value("plot_log_cone_y"))
            if cone_log_y:
                pos_min = min(mL, mM, mS)
                ymin = max(1e-12, pos_min * 0.5) if pos_min > 0 else max(1e-12, ymax * 1e-6)
                ymax_adj = max(ymax, ymin * 10.0)
                dpg.set_axis_limits("plot_cone_ay", ymin, ymax_adj)
            else:
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
    for name in STAT_LAYER_PLOT_ORDER:
        arr = layer_data.get(name)
        if arr is None:
            continue
        slug = name.replace(" ", "_")
        if not dpg.does_item_exist(f"stat_layer_series_{slug}_mean"):
            continue
        flat = np.asarray(arr, dtype=np.float64).ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            continue
        m, s, mn, mx = (
            float(np.mean(flat)),
            float(np.std(flat)),
            float(np.min(flat)),
            float(np.max(flat)),
        )
        for xi, val, metric in zip(
            range(4),
            (m, s, mn, mx),
            ("mean", "std", "min", "max"),
        ):
            t = f"stat_layer_series_{slug}_{metric}"
            if dpg.does_item_exist(t):
                dpg.set_value(t, [[float(xi)], [val]])
        ay_tag = f"stat_layer_ay_{slug}"
        ax_tag = f"stat_layer_ax_{slug}"
        lo = min(m, s, mn, mx)
        hi = max(m, s, mn, mx)
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        if hi <= lo:
            pad = max(abs(lo) * 0.05, 0.05)
            lo, hi = lo - pad, hi + pad
        else:
            pad = (hi - lo) * 0.12
            lo, hi = lo - pad, hi + pad
        if dpg.does_item_exist(ay_tag):
            dpg.set_axis_limits(ay_tag, lo, hi)
        if dpg.does_item_exist(ax_tag):
            dpg.set_axis_limits(ax_tag, -0.5, 3.5)
    # RGC sparkline + histogram: see _update_rgc_dynamics_plots (10 Hz wall clock).
    # Spike raster: updated every frame in the main loop (not throttled with stats).

    _sync_right_panel_plot_axes()


def _resolve_ui_ttf_path() -> str | None:
    """
    Return a path to a .ttf with reliable Latin glyphs for Dear PyGui.

    Prefer Matplotlib's bundled DejaVuSans (no findfont family lookup / no stderr spam).
    Fall back to common OS paths (Arial / DejaVu).
    """
    try:
        import matplotlib as mpl

        bundled = Path(mpl.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        if bundled.is_file():
            return str(bundled)
    except Exception:
        pass
    for p in (
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ):
        if p.is_file():
            return str(p)
    return None


def _load_app_fonts() -> tuple[int | None, int | None]:
    """
    Load main (14px) and side-panel (smaller) fonts from the same .ttf file.
    Returns (main_font_id, side_panel_font_id); either may be None on failure.
    """
    path = _resolve_ui_ttf_path()
    if not path:
        return None, None
    try:
        with dpg.font_registry():
            main = dpg.add_font(path, 14, default_font=True)
            side = dpg.add_font(path, LEFT_PANEL_FONT_PX, default_font=False)
        return main, side
    except Exception:
        return None, None


def _tick_simulation(state: SimState, sim_dt: float) -> None:
    """Advance sim one step, or rebuild mosaic maps only when static-snapshot mode is on."""
    sh = state.config.spatial_heterogeneity
    if (
        state.heterogeneity_dirty
        and sh.mode == SpatialHeterogeneityMode.MOSAIC
        and sh.mosaic.static_snapshot_while_building
    ):
        rebuild_spatial_heterogeneity(state, state.config)
        state.heterogeneity_dirty = False
        return
    tick(state, sim_dt)
    _shared["last_sim_dt"] = float(sim_dt)


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
            _tick_simulation(back, sim_dt)
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
    user_prefs = user_settings.load()
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
    _bind_global_ui_theme(str(user_prefs.get("active_theme_id", "dark_plus")))

    # Load modern font (Inter, SF Pro, Segoe UI, etc.) + smaller variant for left panel
    main_f, side_f = _load_app_fonts()
    _shared["app_font"] = main_f
    _shared["side_panel_font"] = side_f

    # Dynamic texture: large enough for sharp 2D All Layers + single-layer modes
    grid_h, grid_w = state.grid_shape()
    display_w, display_h = _texture_target_hw(state)

    with dpg.texture_registry():
        # Initialize with UI panel background (set in _bind_global_ui_theme)
        empty_tex = np.broadcast_to(
            np.asarray(_ui_letterbox_rgba(), dtype=np.float32),
            (display_h, display_w, 4),
        ).copy()
        dpg.add_dynamic_texture(
            display_w,
            display_h,
            empty_tex.flatten(),
            tag=VIEWPORT_TEX_TAG,
        )

    ctx = AppContext(sim_state=state, textures={"viewport": VIEWPORT_TEX_TAG}, shared=_shared)
    _shared["display_w"] = display_w
    _shared["display_h"] = display_h
    try:
        preset_data = gui_layout.load_preset_dict(
            gui_layout.preset_path(str(user_prefs.get("active_preset", "default")))
        )
    except Exception:
        preset_data = gui_layout.load_preset_dict(gui_layout.preset_path("default"))
    composer = gui_layout.LayoutComposer(preset_data, ctx)
    _shared["layout_composer"] = composer
    _shared["last_export_dir"] = str(user_prefs.get("last_export_dir", str(Path.home())))

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
        composer.ensure_shell("main_window")
        composer.mount_panels()

    if dpg.does_item_exist(VIEWPORT_AREA_TAG):
        tp0 = gui_themes.theme_json_path(str(user_prefs.get("active_theme_id", "dark_plus")))
        if not tp0.is_file():
            tp0 = gui_themes.theme_json_path("dark_plus")
        _refresh_viewport_chrome_theme_from_tokens(gui_themes.load_theme(tp0))
        with dpg.handler_registry(tag=VIEWPORT_ZOOM_WHEEL_REG):
            dpg.add_mouse_wheel_handler(callback=_on_viewport_mouse_wheel)

    if dpg.does_item_exist("layout_preset_combo"):
        dpg.set_value("layout_preset_combo", str(user_prefs.get("active_preset", "default")))
    if dpg.does_item_exist("color_theme_combo"):
        dpg.set_value("color_theme_combo", str(user_prefs.get("active_theme_id", "dark_plus")))
    _shared["active_theme_id"] = str(user_prefs.get("active_theme_id", "dark_plus"))

    _update_stimulus_visibility("spot")  # initial visibility for default type
    _update_view_mode_ui("2D Heatmap")

    # Apply custom font to main window and globally; smaller font for left controls only
    app_font = _shared.get("app_font")
    if app_font is not None:
        dpg.bind_font(app_font)
        dpg.bind_item_font("main_window", app_font)
    side_font = _shared.get("side_panel_font")
    if side_font is not None:
        if dpg.does_item_exist("slot_left_stack"):
            dpg.bind_item_font("slot_left_stack", side_font)
        if dpg.does_item_exist(LEFT_STACK_SCROLL_TAG):
            dpg.bind_item_font(LEFT_STACK_SCROLL_TAG, side_font)
        if dpg.does_item_exist("slot_right_stack"):
            dpg.bind_item_font("slot_right_stack", side_font)

    # About window (hidden by default)
    with dpg.window(label="About", modal=True, show=False, tag="about_window"):
        dpg.add_text("RGC Circuit Simulator - Python")
        dpg.add_text(
            "First-stage human vision simulator: stimulus -> cones -> horizontals -> bipolars "
            "-> amacrines -> RGCs, visualized in layered 2D views."
        )

    dpg.create_viewport(
        title="RGC Circuit Simulator",
        width=WINDOW_SIZE[0],
        height=WINDOW_SIZE[1],
        min_width=composer.min_viewport_client_width(),
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
            _shared["last_export_dir"] = str(Path(path).parent)

    def _on_csv(sender, app_data):
        path = app_data.get("file_path_name")
        if path:
            st = _shared.get("state_front") or _shared.get("state")
            if st is not None:
                export_layer_grids_csv(st, Path(path))
                _shared["last_export_dir"] = str(Path(path).parent)

    def _on_npy(sender, app_data):
        # Directory selector: file_path_name or current_path
        path = app_data.get("file_path_name") or app_data.get("current_path")
        if isinstance(path, (list, tuple)) and path:
            path = path[0]
        if path:
            st = _shared.get("state_front") or _shared.get("state")
            if st is not None:
                export_layer_grids_npy(st, Path(path))
                _shared["last_export_dir"] = str(Path(path))

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
            apply_stimulus_type_change(st, "image")
            if dpg.does_item_exist("stimulus_type_combo"):
                dpg.set_value("stimulus_type_combo", "image")
            _shared["last_export_dir"] = str(Path(path).parent)
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
            _shared["last_export_dir"] = str(Path(path).parent)
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
            _shared["last_export_dir"] = str(Path(path).parent)
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

    _export_default = str(_shared.get("last_export_dir", str(Path.home())))
    with dpg.file_dialog(
        callback=_on_png,
        tag="file_dialog_png",
        show=False,
        modal=True,
        directory_selector=False,
        height=520,
        default_path=_export_default,
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
        default_path=_export_default,
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
        default_path=_export_default,
    ):
        pass

    with dpg.file_dialog(
        callback=_on_stim_image,
        tag="stim_image_dialog",
        show=False,
        modal=True,
        directory_selector=False,
        height=520,
        default_path=_export_default,
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
        default_path=_export_default,
    ):
        pass

    with dpg.file_dialog(
        callback=_on_rec_load,
        tag="file_dialog_rec_load",
        show=False,
        modal=True,
        directory_selector=True,
        height=520,
        default_path=_export_default,
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
    _shared["stats_tick"] = 0
    _shared["all_layers_rgba"] = None  # composite canvas for 2D All Layers view
    _shared["rec_buffer"] = SessionRecordingBuffer()
    _shared["loaded_recording"] = None
    _shared["rec_prev_enabled"] = False
    _shared["playback_active"] = False
    _shared["raster_events"] = deque()
    _shared["last_sim_dt"] = 1.0 / 60.0
    _shared["viewport_zoom"] = 1.0
    _shared["viewport_pan_px"] = [0.0, 0.0]
    _shared["viewport_pan_dragging"] = False
    _shared["_vp_pan_prev_mouse"] = None

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

            view_mode = (
                str(dpg.get_value("view_mode_combo"))
                if dpg.does_item_exist("view_mode_combo")
                else "2D Heatmap"
            )
            base_lw, base_rw = _base_panel_widths_from_preset()
            tex_w = int(_shared.get("display_w", 800))
            tex_h = int(_shared.get("display_h", 600))
            state_layout = _shared.get("state_front")
            if view_mode == "2D All Layers" and state_layout is not None:
                gh, gw = state_layout.grid_shape()
                comp_h, comp_w = gui_layout.composite_pixel_size(gh, gw)
                try:
                    r0 = dpg.get_item_rect_min(VIEWPORT_AREA_TAG)
                    r1 = dpg.get_item_rect_max(VIEWPORT_AREA_TAG)
                    area_h = float(max(240, int(r1[1] - r0[1])))
                except Exception:
                    area_h = max(320.0, float(vh) * 0.82)
                lw, rw, _ = gui_layout.side_widths_for_all_layers_margin(
                    vw,
                    tex_w=tex_w,
                    tex_h=tex_h,
                    comp_h=comp_h,
                    comp_w=comp_w,
                    base_lw=base_lw,
                    base_rw=base_rw,
                    min_viewport_w=MIN_VIEWPORT_WIDTH,
                    area_h=area_h,
                    margin_px=float(ALL_LAYERS_VIEWER_MARGIN_PX),
                )
            else:
                lw, rw = base_lw, base_rw
            _shared["left_panel_width"] = lw
            _shared["right_panel_width"] = rw
            _apply_side_panel_widths_to_ui(lw, rw)

            gui_layout.apply_center_width_from_viewport(vw, vh, _shared)

            # Size and center the image in the middle panel (max space, centered);
            # wheel zoom + Shift+left-drag pan (clamped so the image stays in view).
            if dpg.does_item_exist(VIEWPORT_IMAGE_TAG):
                try:
                    rect_min = dpg.get_item_rect_min(VIEWPORT_AREA_TAG)
                    rect_max = dpg.get_item_rect_max(VIEWPORT_AREA_TAG)
                    area_w = max(1, int(rect_max[0] - rect_min[0]))
                    area_h = max(1, int(rect_max[1] - rect_min[1]))
                except Exception:
                    lw_fb = int(_shared.get("left_panel_width", LEFT_PANEL_WIDTH))
                    rw_fb = int(_shared.get("right_panel_width", RIGHT_PANEL_WIDTH))
                    area_w = max(MIN_VIEWPORT_WIDTH, vw - lw_fb - rw_fb - 32)
                    area_h = max(1, vh - 60)
                _viewport_shift_pan_tick()
                base_scale = min(area_w / max(display_w, 1), area_h / max(display_h, 1))
                zoom = max(
                    VIEWPORT_ZOOM_MIN,
                    min(VIEWPORT_ZOOM_MAX, float(_shared.get("viewport_zoom", 1.0))),
                )
                scale = base_scale * zoom
                img_w = int(display_w * scale)
                img_h = int(display_h * scale)
                base_x = max(0, (area_w - img_w) // 2)
                base_y = max(0, (area_h - img_h) // 2)
                min_px = min(0, area_w - img_w)
                max_px = max(0, area_w - img_w)
                min_py = min(0, area_h - img_h)
                max_py = max(0, area_h - img_h)
                pan = _shared.get("viewport_pan_px", [0.0, 0.0])
                pan_x = float(pan[0]) if len(pan) > 0 else 0.0
                pan_y = float(pan[1]) if len(pan) > 1 else 0.0
                pos_x = int(max(min_px, min(max_px, base_x + pan_x)))
                pos_y = int(max(min_py, min(max_py, base_y + pan_y)))
                _shared["viewport_pan_px"] = [float(pos_x - base_x), float(pos_y - base_y)]
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
                _tick_simulation(state, min(dt, 1.0 / 30.0))
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
        ui_bg = _ui_letterbox_rgba()
        if view_mode == "2D All Layers":
            # No layer-specific overlays or convergence notes in composite mode.
            if dpg.does_item_exist("layer_convergence_note"):
                dpg.set_value("layer_convergence_note", "")
            comp_rgba = _render_all_layers_composite(state)
            rgba = _resize_rgba_to_display(comp_rgba, display_h, display_w, ui_bg)
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
            if layer_name != "Stimulus":
                composite_spatial_heterogeneity_overlays(rgba, state, state.config)
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
            rgba = _resize_rgba_to_display(rgba, display_h, display_w, ui_bg)

        tex_data = np.ascontiguousarray(rgba.astype(np.float32)).flatten()
        img = (rgba * 255).astype(np.uint8)

        _shared["last_frame"] = img
        dpg.set_value(VIEWPORT_TEX_TAG, tex_data)
        # RGC Dynamics: mean-FR trace + histogram (10 Hz wall clock).
        _update_rgc_dynamics_plots(state)
        # Spike raster: every frame so ticks align with simulation time (not stats cadence).
        _update_spike_raster_series(state)
        st = _shared.get("stats_tick", 0)
        _shared["stats_tick"] = st + 1
        if st % _STATS_UI_EVERY_N_FRAMES == 0:
            _update_stats(state)

        dpg.render_dearpygui_frame()

        # Cap frame rate to avoid burning CPU and keep system responsive
        elapsed = time.perf_counter() - frame_start
        sleep_time = (1.0 / TARGET_FPS) - elapsed
        if sleep_time > 0.001:
            time.sleep(sleep_time)

    save_prefs: dict[str, object] = {
        "active_theme_id": str(_shared.get("active_theme_id", user_prefs.get("active_theme_id", "dark_plus"))),
        "active_preset": str(
            dpg.get_value("layout_preset_combo")
            if dpg.does_item_exist("layout_preset_combo")
            else user_prefs.get("active_preset", "default")
        ),
        "last_export_dir": str(_shared.get("last_export_dir", user_prefs.get("last_export_dir", str(Path.home())))),
    }
    user_settings.save(save_prefs)
    dpg.destroy_context()

