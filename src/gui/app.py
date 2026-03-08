from __future__ import annotations

"""
Dear PyGui application wiring together:
- Simulation state + pipeline
- ModernGL 3D viewport
- Control / analysis panels
"""

from pathlib import Path
from typing import Tuple

import dearpygui.dearpygui as dpg
import numpy as np

# Upscale factor for display (texture is grid_resolution * DISPLAY_SCALE)
# Higher = sharper 2D/3D viewers (4 = 1024x1024 for 256 grid)
DISPLAY_SCALE = 4

from src.config import default_config
from src.rendering import RenderContext
from src.rendering.scene_3d.camera import ELEVATION_MAX
from src.rendering.heatmap import grid_to_rgba, spectrum_to_stimulus_rgba
from src.gui.panels.data_export import (
    export_screenshot_png,
    export_layer_grids_csv,
    export_layer_grids_npy,
)
from src.simulation import SimState, tick
from src.simulation.rf_probe import probe_sweep_fast, fit_dog


PANEL_WIDTH = 260
# Minimum size: all three panels (left 260 + center 400 + right 260) must fit
MIN_VIEWPORT_WIDTH = 400
MIN_WINDOW_SIZE: Tuple[int, int] = (MIN_VIEWPORT_WIDTH + 2 * PANEL_WIDTH, 640)  # 920 x 640
# Default size on launch: large enough to see all three panels with no scrolling
WINDOW_SIZE: Tuple[int, int] = (1280, 800)
VIEWPORT_TEX_TAG = "rgc_viewport_tex"
VIEWPORT_AREA_TAG = "viewport_area"
VIEWPORT_IMAGE_TAG = "viewport_image"
DATA_EXPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "exports"

# Shared state for export callbacks, RF compute, and mouse orbit (updated each frame)
_shared: dict = {}


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
        if dpg.does_item_exist("stim_vx"):
            dpg.set_value("stim_vx", state.stimulus_params["vx_deg_s"])
        if dpg.does_item_exist("stim_vy"):
            dpg.set_value("stim_vy", state.stimulus_params["vy_deg_s"])

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
    ]
    # Hide everything first
    hide("stim_radius")
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
        "moving_spot": ["stim_radius", "stim_x_deg", "stim_y_deg", "stim_vx", "stim_vy"],
        "moving_bar": ["stim_x_deg", "stim_y_deg", "stim_orientation", "stim_width", "stim_vx", "stim_vy"],
        "moving_grating": ["stim_x_deg", "stim_y_deg", "stim_orientation", "stim_spatial_freq", "stim_phase", "stim_vx", "stim_vy"],
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
    }
    tags_to_show = show_map.get(stim_type, show_map["spot"])
    for tag in tags_to_show:
        show(tag)


def _update_view_mode_ui(mode: str) -> None:
    """Show 3D-only controls only when 3D view is active."""
    is_3d = mode == "3D Stack"
    if dpg.does_item_exist("camera_3d_node"):
        if is_3d:
            dpg.show_item("camera_3d_node")
        else:
            dpg.hide_item("camera_3d_node")


def _reset_camera(preset: str) -> None:
    ctx = _shared.get("render_ctx")
    if ctx is not None:
        ctx.camera.set_preset(preset)
        for tag, prop, val in [
            ("camera_azimuth", "azimuth", ctx.camera.azimuth),
            ("camera_elevation", "elevation", ctx.camera.elevation),
            ("camera_distance", "distance", ctx.camera.distance),
        ]:
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, val)


def _build_menu_bar() -> None:
    with dpg.menu_bar():
        with dpg.menu(label="File"):
            dpg.add_menu_item(label="Quit", callback=lambda: dpg.stop_dearpygui())
        with dpg.menu(label="View"):
            dpg.add_menu_item(label="Reset camera", callback=lambda: _reset_camera("iso"))
            dpg.add_menu_item(label="Top view", callback=lambda: _reset_camera("top"))
            dpg.add_menu_item(label="Side view", callback=lambda: _reset_camera("side"))
            dpg.add_menu_item(label="Isometric view", callback=lambda: _reset_camera("iso"))
        with dpg.menu(label="Simulation"):
            dpg.add_menu_item(label="Pause / Resume")  # placeholder
        with dpg.menu(label="Help"):
            dpg.add_menu_item(label="About", callback=lambda: dpg.show_item("about_window"))


def _build_left_panel(state: SimState) -> None:
    with dpg.child_window(width=PANEL_WIDTH, height=-1, border=True, autosize_x=False):
        dpg.add_text("View")
        dpg.add_combo(
            label="Mode",
            items=["2D Heatmap", "3D Stack"],
            default_value="2D Heatmap",
            tag="view_mode_combo",
            callback=lambda s, a: _update_view_mode_ui(a),
        )
        dpg.add_combo(
            label="Layer",
            items=[
                "Stimulus",
                "Cones L",
                "Cones M",
                "Cones S",
                "Horizontal",
                "Bipolar ON",
                "Amacrine",
                "RGC Firing (L)",
            ],
            default_value="RGC Firing (L)",
            tag="layer_combo",
        )
        dpg.add_spacer(height=8)
        dpg.add_text("Stimulus")
        with dpg.tree_node(label="Stimulus", default_open=True):
            dpg.add_combo(
                label="Type",
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
                    "dual_spot",
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

        with dpg.tree_node(label="Circuit", default_open=False):
            cfg = state.config
            dpg.add_slider_float(
                label="H alpha (LM)",
                min_value=0.0,
                max_value=1.5,
                default_value=cfg.horizontal.alpha_lm,
                callback=lambda s, a: setattr(cfg.horizontal, "alpha_lm", a),
            )
            dpg.add_slider_float(
                label="Amacrine gamma_AII",
                min_value=0.0,
                max_value=1.5,
                default_value=cfg.amacrine.gamma_aii,
                callback=lambda s, a: setattr(cfg.amacrine, "gamma_aii", a),
            )
            dpg.add_slider_float(
                label="Amacrine gamma_wide",
                min_value=0.0,
                max_value=1.5,
                default_value=cfg.amacrine.gamma_wide,
                callback=lambda s, a: setattr(cfg.amacrine, "gamma_wide", a),
            )

        with dpg.tree_node(label="RGC Params", default_open=False):
            dpg.add_slider_float(
                label="r_max",
                min_value=10.0,
                max_value=300.0,
                default_value=cfg.rgc_nl.r_max,
                callback=lambda s, a: setattr(cfg.rgc_nl, "r_max", a),
            )
            dpg.add_slider_float(
                label="slope",
                min_value=0.5,
                max_value=10.0,
                default_value=cfg.rgc_nl.slope,
                callback=lambda s, a: setattr(cfg.rgc_nl, "slope", a),
            )

        with dpg.tree_node(label="Temporal", default_open=False):
            dpg.add_slider_float(
                label="RGC tau",
                min_value=0.005,
                max_value=0.2,
                default_value=cfg.temporal.rgc_tau,
                callback=lambda s, a: setattr(cfg.temporal, "rgc_tau", a),
            )

        with dpg.tree_node(label="Camera (3D)", default_open=True, tag="camera_3d_node"):
            dpg.add_slider_float(
                label="Azimuth (rad)",
                min_value=-3.14,
                max_value=3.14,
                default_value=0.4,
                tag="camera_azimuth",
                callback=lambda s, a: _shared.get("render_ctx") and setattr(_shared["render_ctx"].camera, "azimuth", a),
            )
            dpg.add_slider_float(
                label="Elevation (rad)",
                min_value=-ELEVATION_MAX,
                max_value=ELEVATION_MAX,
                default_value=0.5,
                tag="camera_elevation",
                callback=lambda s, a: _shared.get("render_ctx") and setattr(_shared["render_ctx"].camera, "elevation", a),
            )
            dpg.add_slider_float(
                label="Distance",
                min_value=2.0,
                max_value=10.0,
                default_value=4.0,
                tag="camera_distance",
                callback=lambda s, a: _shared.get("render_ctx") and setattr(_shared["render_ctx"].camera, "distance", a),
            )
            with dpg.tree_node(label="Layer visibility", default_open=True):
                _layer_names = ["Stimulus", "Cones", "Horizontal", "Bipolar", "Amacrine", "RGC"]
                for name in _layer_names:
                    with dpg.group(horizontal=True):
                        dpg.add_checkbox(label=name, default_value=True, tag=f"layer_vis_{name}",
                            callback=lambda s, a, n=name: _shared.get("render_ctx") and (_shared["render_ctx"].layer_planes.get(n) and setattr(_shared["render_ctx"].layer_planes[n], "visible", a)))
                        dpg.add_slider_float(width=80, min_value=0.0, max_value=1.0, default_value=0.6, tag=f"layer_opacity_{name}",
                            callback=lambda s, a, n=name: _shared.get("render_ctx") and (_shared["render_ctx"].layer_planes.get(n) and setattr(_shared["render_ctx"].layer_planes[n], "opacity", a)))

def _build_right_panel(state: SimState) -> None:
    """Right panel: tab bar so Stats, Export, RF each fit without scrolling."""
    with dpg.child_window(width=PANEL_WIDTH, height=-1, border=True, autosize_x=False):
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
            with dpg.tab(label="Export"):
                dpg.add_button(label="Save screenshot (PNG)", tag="btn_export_png", callback=lambda: dpg.show_item("file_dialog_png"))
                dpg.add_button(label="Save layer stats (CSV)", tag="btn_export_csv", callback=lambda: dpg.show_item("file_dialog_csv"))
                dpg.add_button(label="Save layer grids (.npy)", tag="btn_export_npy", callback=lambda: dpg.show_item("file_dialog_npy"))
            with dpg.tab(label="Receptive Field"):
                dpg.add_combo(label="RGC type", items=["midget_on_L", "midget_off_L", "parasol_on", "parasol_off"], default_value="midget_on_L", tag="rf_rgc_type")
                dpg.add_button(label="Compute RF (24x24 sweep)", tag="btn_compute_rf", callback=lambda: _shared.update({"rf_pending": True}))
                dpg.add_text("sigma_c: -  sigma_s: -  ratio: -", tag="rf_dog_result")


def _build_center_viewport(display_width: int, display_height: int) -> None:
    """Center panel: displays the simulation heatmap via a dynamic texture."""
    with dpg.child_window(border=True, width=-1, height=-1, tag=VIEWPORT_AREA_TAG):
        # Image tag so we can resize/center each frame; initial size placeholder
        dpg.add_image(VIEWPORT_TEX_TAG, tag=VIEWPORT_IMAGE_TAG, width=400, height=400)


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
    # RGC histogram
    if state.fr_midget_on_L is not None and dpg.does_item_exist("hist_series"):
        flat = state.fr_midget_on_L.flatten()
        flat = flat[np.isfinite(flat)]
        if len(flat) > 0:
            bins = 16
            counts, edges = np.histogram(flat, bins=bins)
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
    })

    dpg.create_context()

    # Load modern font (Inter, SF Pro, Segoe UI, etc.)
    _shared["app_font"] = _load_app_font()

    # Texture registry: display at DISPLAY_SCALE × grid resolution for visibility
    grid_h, grid_w = state.grid_shape()
    display_w, display_h = grid_w * DISPLAY_SCALE, grid_h * DISPLAY_SCALE

    with dpg.texture_registry():
        # Initialize with dark gray (matches 3D clear color) to avoid flashing on load
        empty_tex = np.full((display_h, display_w, 4), [0.02, 0.02, 0.04, 1.0], dtype=np.float32)
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
    _update_view_mode_ui("2D Heatmap")   # hide 3D-only controls until 3D is selected

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
            "-> amacrines -> RGCs, visualized in 3D."
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
    # Force viewport to default size so all three panels are visible on launch
    dpg.configure_viewport(0, width=WINDOW_SIZE[0], height=WINDOW_SIZE[1])

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
            export_layer_grids_csv(state, Path(path))

    def _on_npy(sender, app_data):
        # Directory selector: file_path_name or current_path
        path = app_data.get("file_path_name") or app_data.get("current_path")
        if isinstance(path, (list, tuple)) and path:
            path = path[0]
        if path:
            export_layer_grids_npy(state, Path(path))

    with dpg.file_dialog(
        callback=_on_png,
        tag="file_dialog_png",
        show=False,
        modal=True,
        directory_selector=False,
    ):
        dpg.add_file_extension(".*")
        dpg.add_file_extension(".png", color=(0, 255, 0, 255))

    with dpg.file_dialog(
        callback=_on_csv,
        tag="file_dialog_csv",
        show=False,
        modal=True,
        directory_selector=False,
    ):
        dpg.add_file_extension(".*")
        dpg.add_file_extension(".csv", color=(0, 255, 0, 255))

    with dpg.file_dialog(
        callback=_on_npy,
        tag="file_dialog_npy",
        show=False,
        modal=True,
        directory_selector=True,
    ):
        pass

    # Shared state for main loop
    _shared["state"] = state
    _shared["last_frame"] = None
    _shared["rf_pending"] = False
    _shared["render_ctx"] = None
    _shared["last_mouse_pos"] = None  # for 3D orbit
    _shared["wheel_delta"] = 0  # accumulated scroll (consumed each frame when viewport hovered)
    _shared["frame_count"] = 0  # for deferred resize at startup
    _shared["rgc_fr_history"] = []  # for sparkline (last 100 ticks)

    # Mouse wheel handler for 3D zoom (DPG has no get_mouse_wheel polling)
    def _on_wheel(sender, app_data):
        # app_data can be scalar (vertical) or (x, y) tuple
        delta = app_data if isinstance(app_data, (int, float)) else (app_data[1] if isinstance(app_data, (list, tuple)) and len(app_data) > 1 else app_data)
        _shared["wheel_delta"] = _shared.get("wheel_delta", 0) + float(delta)

    with dpg.handler_registry():
        dpg.add_mouse_wheel_handler(callback=_on_wheel)

    # Main loop: step simulation, render via 2D or 3D, blit into DPG dynamic texture.
    last_time = dpg.get_total_time()
    while dpg.is_dearpygui_running():
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
            center_w = max(MIN_VIEWPORT_WIDTH, vw - 2 * PANEL_WIDTH - slack)
            dpg.configure_item(VIEWPORT_AREA_TAG, width=center_w)

            # Size and center the heatmap/3D image in the middle panel (max space, centered)
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

        # Step simulation
        tick(state, dt)

        # RF compute (runs probe sweep, can be slow)
        if _shared.get("rf_pending"):
            _shared["rf_pending"] = False
            rgc_type = dpg.get_value("rf_rgc_type") if dpg.does_item_exist("rf_rgc_type") else "midget_on_L"
            try:
                x_deg, y_deg, rf_map = probe_sweep_fast(state, rgc_type=rgc_type, probe_resolution=24)
                dog = fit_dog(x_deg, y_deg, rf_map)
                ratio = dog.sigma_surround / dog.sigma_center if dog.sigma_center > 1e-9 else 0
                txt = f"sigma_c: {dog.sigma_center:.4f}  sigma_s: {dog.sigma_surround:.4f}  ratio: {ratio:.2f}"
                if dpg.does_item_exist("rf_dog_result"):
                    dpg.set_value("rf_dog_result", txt)
            except Exception as e:
                if dpg.does_item_exist("rf_dog_result"):
                    dpg.set_value("rf_dog_result", f"Error: {e}")

        # Render: 2D heatmap or 3D stack
        view_mode = dpg.get_value("view_mode_combo") if dpg.does_item_exist("view_mode_combo") else "2D Heatmap"
        if view_mode == "3D Stack":
            if _shared.get("render_ctx") is None:
                _shared["render_ctx"] = RenderContext(
                    size=(display_w, display_h),
                    config=cfg,
                )
            ctx = _shared["render_ctx"]
            # Ensure planes exist before syncing visibility (creates on first 3D render)
            ctx.ensure_scene(state)
            # Sync layer visibility/opacity from UI
            for name in ["Stimulus", "Cones", "Horizontal", "Bipolar", "Amacrine", "RGC"]:
                plane = ctx.layer_planes.get(name)
                if plane is not None:
                    if dpg.does_item_exist(f"layer_vis_{name}"):
                        plane.visible = dpg.get_value(f"layer_vis_{name}")
                    if dpg.does_item_exist(f"layer_opacity_{name}"):
                        plane.opacity = dpg.get_value(f"layer_opacity_{name}")
            # Camera damping
            ctx.camera.integrate(dt)
            # Mouse orbit: drag to rotate, scroll to zoom (only when viewport hovered)
            if dpg.does_item_exist(VIEWPORT_AREA_TAG) and dpg.is_item_hovered(VIEWPORT_AREA_TAG):
                pos = dpg.get_mouse_pos()
                if dpg.is_mouse_button_down(0):
                    last = _shared.get("last_mouse_pos")
                    if last is not None:
                        dx, dy = pos[0] - last[0], pos[1] - last[1]
                        # Reduced sensitivity for smoother orbit
                        ctx.camera.azimuth -= dx * 0.0018
                        ctx.camera.elevation = max(-ELEVATION_MAX, min(ELEVATION_MAX, ctx.camera.elevation + dy * 0.0018))
                        ctx.camera._azimuth_vel -= dx * 0.0006
                        ctx.camera._elevation_vel += dy * 0.0006
                    _shared["last_mouse_pos"] = (pos[0], pos[1])
                else:
                    _shared["last_mouse_pos"] = None
                wheel = _shared.get("wheel_delta", 0)
                if wheel != 0:
                    # Gentler zoom steps
                    factor = 0.96 if wheel > 0 else 1.04
                    ctx.camera.distance = max(2.0, min(12.0, ctx.camera.distance * factor))
                    _shared["wheel_delta"] = 0
            else:
                _shared["last_mouse_pos"] = None
            img = ctx.render_3d(state)
            # 3D returns uint8; DPG wants float 0-1
            tex_data = (img.astype(np.float32) / 255.0).flatten()
        else:
            # 2D heatmap
            layer_name = dpg.get_value("layer_combo") if dpg.does_item_exist("layer_combo") else "RGC Firing (L)"
            if layer_name == "Stimulus" and state.stimulus_spectrum is not None:
                wl = state.config.spectral.wavelengths
                rgba = spectrum_to_stimulus_rgba(state.stimulus_spectrum, wl)
            else:
                layer_map = {
                    "Cones L": (state.cone_L, "firing"),
                    "Cones M": (state.cone_M, "firing"),
                    "Cones S": (state.cone_S, "firing"),
                    "Horizontal": (state.h_activation, "firing"),
                    "Bipolar ON": (state.bp_diffuse_on, "firing"),
                    "Amacrine": (state.amacrine_aii, "firing"),
                    "RGC Firing (L)": (state.fr_midget_on_L, "firing"),
                }
                layer, colormap = layer_map.get(layer_name, (state.fr_midget_on_L, "firing"))
                if layer is None:
                    layer = np.zeros(state.grid_shape(), dtype=np.float32)
                rgba = grid_to_rgba(layer, colormap=colormap)
            # Upscale to match texture size (display_w × display_h) to avoid flashing
            rgba = np.repeat(np.repeat(rgba, DISPLAY_SCALE, axis=0), DISPLAY_SCALE, axis=1)
            tex_data = np.ascontiguousarray(rgba.astype(np.float32)).flatten()
            img = (rgba * 255).astype(np.uint8)

        _shared["last_frame"] = img
        dpg.set_value(VIEWPORT_TEX_TAG, tex_data)
        _update_stats(state)

        dpg.render_dearpygui_frame()

    dpg.destroy_context()

