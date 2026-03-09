from __future__ import annotations

"""
Dear PyGui application wiring together:
- Simulation state + pipeline
- ModernGL 3D viewport
- Control / analysis panels
"""

import os
import random
import threading
import time
from pathlib import Path
from typing import Tuple

# Run simulation on main thread (no background worker) for smoother 60 FPS; set SIM_ON_MAIN_THREAD=1
SIM_ON_MAIN_THREAD = os.environ.get("SIM_ON_MAIN_THREAD", "").strip().lower() in ("1", "true", "yes")
# Tick every Nth frame when on main thread to hit 60 FPS (SIM_TICK_EVERY_N=2 → 30 Hz sim)
SIM_TICK_EVERY_N = max(1, int(os.environ.get("SIM_TICK_EVERY_N", "1")))

# Cap main loop at this FPS to reduce CPU use and keep UI responsive
TARGET_FPS = 60

import dearpygui.dearpygui as dpg
import numpy as np
from PIL import Image

# Upscale factor for display (texture is grid_resolution * DISPLAY_SCALE)
# 4 = 1024x1024 for a 256 grid: good balance of sharpness vs speed.
DISPLAY_SCALE = 4

from src.config import default_config
from src.rendering.scene_3d.camera import ELEVATION_MAX, DEFAULT_AZIMUTH, DEFAULT_ELEVATION
from src.rendering.heatmap import (
    block_average_downsample,
    block_average_downsample_rgba,
    draw_scale_bar_rgba,
    grid_to_rgba,
    spectrum_to_stimulus_rgba,
)
from src.rendering.overlay import draw_cell_overlay
from src.simulation.cell_positions import (
    CellPositions,
    pick_nearest_cell_any_layer,
)
from src.simulation.connectivity import (
    ConnectivityCache,
    RGCConnectivityResult,
    compute_amacrine_connectivity,
    compute_bipolar_connectivity,
    compute_cone_connectivity,
    compute_horizontal_connectivity,
    compute_rgc_connectivity,
)
from src.gui.panels.cell_inspector import update_inspector
from src.viewers.viewer_3d import HAS_VISPY, VispyViewer3D
from src.gui.panels.data_export import (
    export_screenshot_png,
    export_layer_grids_csv,
    export_layer_grids_npy,
)
from src.simulation import SimState, tick
from src.simulation.rf_probe import probe_sweep_fast, fit_dog
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
}


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

# Max display resolution for 2D viewer (block-average downsampling above this)
MAX_DISPLAY_SIDE = 1024


def _ensure_cell_positions(state: SimState) -> CellPositions:
    """Build or return cached CellPositions for the current grid. Uses cKDTree for O(log N) picks."""
    cp = _shared.get("cell_positions")
    grid_size = state.grid_shape()[0]
    if cp is not None and cp.grid_size == grid_size:
        return cp
    cfg = state.config.retina
    fovea = (grid_size / 2.0, grid_size / 2.0)
    cp = CellPositions(
        grid_size=grid_size,
        microns_per_px=cfg.microns_per_px,
        fovea_center=fovea,
    )
    cone_subsample = 4 if grid_size > 512 else 1  # fewer cone points for huge grids
    cp.init_default(cone_subsample=cone_subsample)
    _shared["cell_positions"] = cp
    if "connectivity_cache" not in _shared:
        _shared["connectivity_cache"] = ConnectivityCache()
    return cp


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
        "moving_spot": ["stim_radius", "stim_x_deg", "stim_y_deg", "stim_vx", "stim_vy"],
        "moving_bar": ["stim_x_deg", "stim_y_deg", "stim_orientation", "stim_width", "stim_vx", "stim_vy"],
        "moving_grating": ["stim_x_deg", "stim_y_deg", "stim_orientation", "stim_spatial_freq", "stim_phase", "stim_vx", "stim_vy"],
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
    """Show 3D-only controls when 3D view is active; hide Pick layer combo in 2D."""
    is_3d = mode == "3D Stack"
    if dpg.does_item_exist("camera_3d_node"):
        if is_3d:
            dpg.show_item("camera_3d_node")
        else:
            dpg.hide_item("camera_3d_node")
    if dpg.does_item_exist("pick_layer_combo"):
        if is_3d:
            dpg.show_item("pick_layer_combo")
        else:
            dpg.hide_item("pick_layer_combo")
    if dpg.does_item_exist("viewer3d_toolbar"):
        if is_3d:
            dpg.show_item("viewer3d_toolbar")
        else:
            dpg.hide_item("viewer3d_toolbar")


def _reset_camera(preset: str) -> None:
    viewer = _shared.get("vispy_viewer")
    if viewer is not None and HAS_VISPY:
        cam = viewer._camera
        if preset == "iso":
            cam.azimuth, cam.elevation, cam.distance = 45.0, 20.0, 12.0
        elif preset == "top":
            cam.azimuth, cam.elevation, cam.distance = 0.0, -89.0, 12.0
        elif preset == "front":
            cam.azimuth, cam.elevation, cam.distance = 0.0, 0.0, 12.0
        else:
            cam.azimuth, cam.elevation, cam.distance = 45.0, 20.0, 12.0
        if dpg.does_item_exist("camera_azimuth"):
            dpg.set_value("camera_azimuth", float(cam.azimuth))
        if dpg.does_item_exist("camera_elevation"):
            dpg.set_value("camera_elevation", float(cam.elevation))
        if dpg.does_item_exist("camera_distance"):
            dpg.set_value("camera_distance", float(cam.distance))


def _build_menu_bar() -> None:
    with dpg.menu_bar():
        with dpg.menu(label="File"):
            dpg.add_menu_item(label="Quit", callback=lambda: dpg.stop_dearpygui())
        with dpg.menu(label="View"):
            dpg.add_menu_item(label="Reset camera", callback=lambda: _reset_camera("iso"))
            dpg.add_menu_item(label="Top view", callback=lambda: _reset_camera("top"))
            dpg.add_menu_item(label="Front view", callback=lambda: _reset_camera("front"))
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
        dpg.add_combo(
            label="Pick layer (click viewport to inspect)",
            items=["RGC", "Cone", "Bipolar", "Horizontal", "Amacrine"],
            default_value="RGC",
            tag="pick_layer_combo",
        )
        dpg.add_checkbox(
            label="Biological scale (weight by convergence)",
            default_value=False,
            tag="biological_scale_2d",
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
            dpg.add_checkbox(label="Show signal flow", default_value=True, tag="show_signal_flow")
            dpg.add_slider_float(label="Slice position", min_value=-0.5, max_value=0.5, default_value=0.0,
                tag="slice_position")
            dpg.add_text("Connectivity types")
            dpg.add_checkbox(label="Cone -> Horizontal", default_value=True, tag="show_cone_to_horizontal")
            dpg.add_checkbox(label="Cone -> Bipolar", default_value=True, tag="show_cone_to_bipolar")
            dpg.add_checkbox(label="Bipolar -> Amacrine", default_value=True, tag="show_bipolar_to_amacrine")
            dpg.add_checkbox(label="Bipolar -> RGC", default_value=True, tag="show_bipolar_to_rgc")
            dpg.add_combo(
                label="Fovea / Periphery",
                items=["Fovea (~1:1 cone→RGC)", "Periphery (up to ~30:1)"],
                default_value="Fovea (~1:1 cone→RGC)",
                tag="fovea_periphery_combo",
            )
            dpg.add_slider_float(
                label="Azimuth (rad)",
                min_value=-3.14,
                max_value=3.14,
                default_value=float(DEFAULT_AZIMUTH),
                tag="camera_azimuth",
                callback=lambda s, a: _shared.get("vispy_viewer") and setattr(_shared["vispy_viewer"]._camera, "azimuth", a),
            )
            dpg.add_slider_float(
                label="Elevation (rad)",
                min_value=-ELEVATION_MAX,
                max_value=ELEVATION_MAX,
                default_value=float(DEFAULT_ELEVATION),
                tag="camera_elevation",
                callback=lambda s, a: _shared.get("vispy_viewer") and setattr(_shared["vispy_viewer"]._camera, "elevation", a),
            )
            dpg.add_slider_float(
                label="Distance",
                min_value=3.0,
                max_value=14.0,
                default_value=6.0,
                tag="camera_distance",
                callback=lambda s, a: _shared.get("vispy_viewer") and setattr(_shared["vispy_viewer"]._camera, "distance", a),
            )
            with dpg.tree_node(label="Layer visibility", default_open=True):
                _layer_names = ["Stimulus", "Cones", "Horizontal", "Bipolar", "Amacrine", "RGC"]
                for name in _layer_names:
                    with dpg.group(horizontal=True):
                        dpg.add_checkbox(label=name, default_value=True, tag=f"layer_vis_{name}")
                        dpg.add_slider_float(width=80, min_value=0.0, max_value=1.0, default_value=0.85, tag=f"layer_opacity_{name}")

def _set_conn_weight(state: SimState, key: str, value: float) -> None:
    if hasattr(state.config, "connectivity_weights"):
        setattr(state.config.connectivity_weights, key, max(0.0, min(3.0, value)))


def _set_connectivity_dirty() -> None:
    _shared["connectivity_dirty"] = True


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
        setattr(cw, key, random.uniform(0.5, 2.0))
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
                dpg.add_button(label="Save screenshot (PNG)", width=-1, tag="btn_export_png", callback=lambda: dpg.show_item("file_dialog_png"))
                dpg.add_button(label="Save layer stats (CSV)", width=-1, tag="btn_export_csv", callback=lambda: dpg.show_item("file_dialog_csv"))
                dpg.add_button(label="Save layer grids (.npy)", width=-1, tag="btn_export_npy", callback=lambda: dpg.show_item("file_dialog_npy"))
            with dpg.tab(label="Connectivity"):
                dpg.add_text("Weight matrix (0.0–3.0). Changes apply to pipeline and 3D lines.")
                cw = state.config.connectivity_weights
                dpg.add_input_float(label="Cone -> Horizontal", default_value=cw.cone_to_horizontal, min_value=0.0, max_value=3.0, min_clamped=True, max_clamped=True, width=60, tag="conn_cone_to_horizontal",
                    callback=lambda s, a: (_set_conn_weight(state, "cone_to_horizontal", a), _set_connectivity_dirty()))
                dpg.add_input_float(label="Cone -> Bipolar", default_value=cw.cone_to_bipolar, min_value=0.0, max_value=3.0, min_clamped=True, max_clamped=True, width=60, tag="conn_cone_to_bipolar",
                    callback=lambda s, a: (_set_conn_weight(state, "cone_to_bipolar", a), _set_connectivity_dirty()))
                dpg.add_input_float(label="Horizontal -> Cone", default_value=cw.horizontal_to_cone, min_value=0.0, max_value=3.0, min_clamped=True, max_clamped=True, width=60, tag="conn_horizontal_to_cone",
                    callback=lambda s, a: (_set_conn_weight(state, "horizontal_to_cone", a), _set_connectivity_dirty()))
                dpg.add_input_float(label="Bipolar -> Amacrine", default_value=cw.bipolar_to_amacrine, min_value=0.0, max_value=3.0, min_clamped=True, max_clamped=True, width=60, tag="conn_bipolar_to_amacrine",
                    callback=lambda s, a: (_set_conn_weight(state, "bipolar_to_amacrine", a), _set_connectivity_dirty()))
                dpg.add_input_float(label="Amacrine -> Bipolar", default_value=cw.amacrine_to_bipolar, min_value=0.0, max_value=3.0, min_clamped=True, max_clamped=True, width=60, tag="conn_amacrine_to_bipolar",
                    callback=lambda s, a: (_set_conn_weight(state, "amacrine_to_bipolar", a), _set_connectivity_dirty()))
                dpg.add_input_float(label="Bipolar -> RGC", default_value=cw.bipolar_to_rgc, min_value=0.0, max_value=3.0, min_clamped=True, max_clamped=True, width=60, tag="conn_bipolar_to_rgc",
                    callback=lambda s, a: (_set_conn_weight(state, "bipolar_to_rgc", a), _set_connectivity_dirty()))
                dpg.add_button(label="Reset to defaults", tag="conn_reset", callback=lambda: _reset_connectivity_weights(state))
                dpg.add_button(label="Randomize", tag="conn_randomize", callback=lambda: _randomize_connectivity_weights(state))
            with dpg.tab(label="Inspector"):
                from src.gui.panels.cell_inspector import build_inspector_panel
                build_inspector_panel()
            with dpg.tab(label="Cell Params"):
                cfg = state.config
                with dpg.tree_node(label="Midget RGC", default_open=True):
                    dpg.add_input_float(
                        label="Dendritic sigma (deg)",
                        default_value=cfg.dendritic.sigma_midget_deg,
                        step=0.001,
                        callback=lambda s, a: setattr(cfg.dendritic, "sigma_midget_deg", a),
                    )
                    dpg.add_input_float(
                        label="Max firing (sp/s)",
                        default_value=cfg.rgc_nl.r_max,
                        step=1.0,
                        callback=lambda s, a: setattr(cfg.rgc_nl, "r_max", a),
                    )
                    dpg.add_input_float(
                        label="LN slope",
                        default_value=cfg.rgc_nl.slope,
                        step=0.1,
                        callback=lambda s, a: setattr(cfg.rgc_nl, "slope", a),
                    )
                    dpg.add_input_float(
                        label="LN half-point",
                        default_value=cfg.rgc_nl.x_half,
                        step=0.01,
                        callback=lambda s, a: setattr(cfg.rgc_nl, "x_half", a),
                    )
                    dpg.add_input_float(
                        label="Tau (s)",
                        default_value=cfg.temporal.rgc_tau,
                        step=0.01,
                        callback=lambda s, a: setattr(cfg.temporal, "rgc_tau", a),
                    )
                with dpg.tree_node(label="Parasol RGC", default_open=False):
                    dpg.add_input_float(
                        label="Dendritic sigma (deg)",
                        default_value=cfg.dendritic.sigma_parasol_deg,
                        step=0.001,
                        callback=lambda s, a: setattr(cfg.dendritic, "sigma_parasol_deg", a),
                    )
                    dpg.add_input_float(
                        label="Max firing (sp/s)",
                        default_value=cfg.rgc_nl.r_max,
                        step=1.0,
                        callback=lambda s, a: setattr(cfg.rgc_nl, "r_max", a),
                    )
                with dpg.tree_node(label="Bipolar (midget/diffuse)", default_open=False):
                    dpg.add_input_float(
                        label="Sigma diffuse (deg)",
                        default_value=cfg.bipolar.sigma_diffuse_deg,
                        step=0.001,
                        callback=lambda s, a: setattr(cfg.bipolar, "sigma_diffuse_deg", a),
                    )
                    dpg.add_input_float(
                        label="Tau (s)",
                        default_value=cfg.temporal.bipolar_tau,
                        step=0.01,
                        callback=lambda s, a: setattr(cfg.temporal, "bipolar_tau", a),
                    )
                with dpg.tree_node(label="Horizontal Cells", default_open=False):
                    dpg.add_input_float(
                        label="Sigma LM (deg)",
                        default_value=cfg.horizontal.sigma_lm_deg,
                        step=0.001,
                        callback=lambda s, a: setattr(cfg.horizontal, "sigma_lm_deg", a),
                    )
                    dpg.add_input_float(
                        label="Sigma S (deg)",
                        default_value=cfg.horizontal.sigma_s_deg,
                        step=0.001,
                        callback=lambda s, a: setattr(cfg.horizontal, "sigma_s_deg", a),
                    )
                    dpg.add_input_float(
                        label="Alpha LM",
                        default_value=cfg.horizontal.alpha_lm,
                        step=0.01,
                        callback=lambda s, a: setattr(cfg.horizontal, "alpha_lm", a),
                    )
                    dpg.add_input_float(
                        label="Alpha S",
                        default_value=cfg.horizontal.alpha_s,
                        step=0.01,
                        callback=lambda s, a: setattr(cfg.horizontal, "alpha_s", a),
                    )
                    dpg.add_input_float(
                        label="Tau (s)",
                        default_value=cfg.temporal.horizontal_tau,
                        step=0.01,
                        callback=lambda s, a: setattr(cfg.temporal, "horizontal_tau", a),
                    )
                with dpg.tree_node(label="Amacrine (AII)", default_open=False):
                    dpg.add_input_float(
                        label="Sigma (deg)",
                        default_value=cfg.amacrine.sigma_aii_deg,
                        step=0.001,
                        callback=lambda s, a: setattr(cfg.amacrine, "sigma_aii_deg", a),
                    )
                    dpg.add_input_float(
                        label="Gamma (weight)",
                        default_value=cfg.amacrine.gamma_aii,
                        step=0.01,
                        callback=lambda s, a: setattr(cfg.amacrine, "gamma_aii", a),
                    )
                    dpg.add_input_float(
                        label="Tau (s)",
                        default_value=cfg.temporal.amacrine_tau,
                        step=0.01,
                        callback=lambda s, a: setattr(cfg.temporal, "amacrine_tau", a),
                    )
                with dpg.tree_node(label="Amacrine (Wide-field)", default_open=False):
                    dpg.add_input_float(
                        label="Sigma (deg)",
                        default_value=cfg.amacrine.sigma_wide_deg,
                        step=0.001,
                        callback=lambda s, a: setattr(cfg.amacrine, "sigma_wide_deg", a),
                    )
                    dpg.add_input_float(
                        label="Gamma (weight)",
                        default_value=cfg.amacrine.gamma_wide,
                        step=0.01,
                        callback=lambda s, a: setattr(cfg.amacrine, "gamma_wide", a),
                    )
                with dpg.tree_node(label="3D Viewer Display", default_open=False):
                    from src.simulation.bio_constants import CONE_FRAC_L, CONE_FRAC_M, CONE_FRAC_S, ROD_CONE_RATIO

                    dpg.add_text(f"L cone fraction: {float(CONE_FRAC_L):.3f}")
                    dpg.add_text(f"M cone fraction: {float(CONE_FRAC_M):.3f}")
                    dpg.add_text(f"S cone fraction: {float(CONE_FRAC_S):.3f}")
                    dpg.add_text(f"Rod : Cone ratio: {float(ROD_CONE_RATIO):.1f}")
                dpg.add_spacer(height=8)
                dpg.add_button(label="Reset to defaults", width=-1)
                dpg.add_button(label="Save preset", width=-1)
                dpg.add_button(label="Load preset", width=-1)
            with dpg.tab(label="Receptive Field"):
                dpg.add_combo(label="RGC type", items=["midget_on_L", "midget_off_L", "parasol_on", "parasol_off"], default_value="midget_on_L", tag="rf_rgc_type")
                dpg.add_button(label="Compute RF (24x24 sweep)", tag="btn_compute_rf", callback=lambda: _shared.update({"rf_pending": True}))
                dpg.add_text("sigma_c: -  sigma_s: -  ratio: -", tag="rf_dog_result")


def _build_center_viewport(display_width: int, display_height: int) -> None:
    """Center panel: displays the simulation heatmap or 3D stack plus 3D toolbar."""
    with dpg.child_window(border=True, width=-1, height=-1, tag=VIEWPORT_AREA_TAG):
        with dpg.group(horizontal=False):
            # Image tag so we can resize/center each frame; initial size placeholder
            dpg.add_image(VIEWPORT_TEX_TAG, tag=VIEWPORT_IMAGE_TAG, width=400, height=400)
            # Bottom 3D toolbar (shown only in 3D mode).
            with dpg.group(horizontal=True, tag="viewer3d_toolbar"):
                dpg.add_button(label="Zoom -", width=60, callback=lambda: _shared.get("vispy_viewer") and HAS_VISPY and _shared["vispy_viewer"].add_zoom(+0.5))
                dpg.add_slider_float(
                    label="",
                    width=140,
                    min_value=3.0,
                    max_value=50.0,
                    default_value=12.0,
                    tag="viewer3d_zoom_slider",
                    callback=lambda s, a: _shared.get("vispy_viewer") and HAS_VISPY and setattr(_shared["vispy_viewer"]._camera, "distance", float(a)),
                )
                dpg.add_button(label="Zoom +", width=60, callback=lambda: _shared.get("vispy_viewer") and HAS_VISPY and _shared["vispy_viewer"].add_zoom(-0.5))
                dpg.add_spacer(width=8)
                dpg.add_button(label="⟲ Reset", width=70, callback=lambda: _reset_camera("iso"))
                dpg.add_checkbox(
                    label="Flat / Sphere",
                    tag="viewer3d_flat_sphere",
                    default_value=False,
                )
                dpg.add_spacer(width=8)
                dpg.add_text("Cells:")
                dpg.add_slider_int(
                    label="",
                    width=130,
                    min_value=1000,
                    max_value=50000,
                    default_value=8000,
                    tag="viewer3d_cells",
                )
                dpg.add_checkbox(label="Max density", tag="viewer3d_max_density", default_value=False)


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
            mn, mx = float(np.min(flat)), float(np.max(flat))
            if mx <= mn:
                mx = mn + 1.0  # avoid "too many bins for data range"
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
        if not path:
            return
        try:
            # Keep RGB so that colors can be binned by L/M/S.
            img = Image.open(path).convert("RGB")
            h, w = state.grid_shape()
            img = img.resize((w, h), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32)
            # Store in 0–1 so spectral construction can preserve RGB ratios.
            state.stimulus_params["image_mask"] = (arr / 255.0).astype(np.float32)
        except Exception as e:
            # Fallback: simple stderr print so the app keeps running.
            print(f"Failed to load stimulus image: {e}")

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

    # Shared state for main loop (double-buffer when worker used; else single state)
    _shared["state"] = state  # legacy alias
    _shared["state_front"] = state
    _shared["state_back"] = state_back
    _shared["sim_on_main_thread"] = SIM_ON_MAIN_THREAD
    _shared["sim_tick_every_n"] = SIM_TICK_EVERY_N
    _shared["sim_tick_counter"] = 0
    _shared["last_frame"] = None
    _shared["rf_pending"] = False
    _shared["vispy_viewer"] = None
    _shared["connectivity_dirty"] = False
    _shared["last_mouse_pos"] = None  # for 3D orbit
    _shared["wheel_delta"] = 0  # accumulated scroll (consumed each frame when viewport hovered)
    _shared["frame_count"] = 0  # for deferred resize at startup
    _shared["rgc_fr_history"] = []  # for sparkline (last 100 ticks)
    _shared["picked_cell"] = None  # (layer_name, cell_id, connectivity_result) or None
    _shared["mouse_was_down"] = False  # for pick click detection
    _shared["mouse_down_pos"] = None  # (x, y) when button went down; used to avoid pick on 3D drag
    _shared["stats_tick"] = 0  # throttle stats update
    _shared["debug_3d_prints"] = 0  # limit debug logging for 3D frames

    if not SIM_ON_MAIN_THREAD and state_back is not None:
        threading.Thread(target=_sim_worker, daemon=True).start()

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

        # Use display buffer (worker writes here) or single state (sim on main thread)
        if _shared.get("sim_on_main_thread"):
            state = _shared["state_front"]
            _shared["state"] = state
            # Tick on main thread every Nth frame to keep 60 FPS
            ctr = _shared.get("sim_tick_counter", 0)
            _shared["sim_tick_counter"] = (ctr + 1) % _shared.get("sim_tick_every_n", 1)
            if ctr == 0:
                tick(state, min(dt, 1.0 / 30.0))
        else:
            state = _shared["state_front"]
            _shared["state"] = state

        # Click on viewport (2D or 3D) to select cell and show connectivity in Inspector
        mouse_down_now = dpg.is_mouse_button_down(0)
        if mouse_down_now and not _shared.get("mouse_was_down"):
            _shared["mouse_down_pos"] = dpg.get_mouse_pos()
        if _shared.get("mouse_was_down") and not mouse_down_now and dpg.is_item_hovered(VIEWPORT_IMAGE_TAG):
            # Only count as pick if mouse barely moved (avoid treating 3D camera drag as pick)
            down_pos = _shared.get("mouse_down_pos")
            mx, my = dpg.get_mouse_pos()
            is_drag = False
            if down_pos is not None:
                dx, dy = mx - down_pos[0], my - down_pos[1]
                if dx * dx + dy * dy > 64:  # moved more than 8 px → was a drag
                    is_drag = True
            if not is_drag:
                try:
                    rect_min = dpg.get_item_rect_min(VIEWPORT_IMAGE_TAG)
                    rect_max = dpg.get_item_rect_max(VIEWPORT_IMAGE_TAG)
                    mx, my = dpg.get_mouse_pos()
                    local_x = mx - rect_min[0]
                    local_y = my - rect_min[1]
                    img_w = max(1, rect_max[0] - rect_min[0])
                    img_h = max(1, rect_max[1] - rect_min[1])
                    if 0 <= local_x < img_w and 0 <= local_y < img_h:
                        cp = _ensure_cell_positions(state)
                        grid_h, grid_w = state.grid_shape()
                        grid_x = local_x / img_w * grid_w
                        grid_y = local_y / img_h * grid_h
                        pick_radius_px = max(20.0, grid_w / 64.0)  # scale with grid so large fields are pickable
                        pick_layer, cell_id = pick_nearest_cell_any_layer(cp, grid_x, grid_y, pick_radius_px)
                        cache = _shared.get("connectivity_cache")
                        result = None
                        if pick_layer is not None and cell_id is not None and cache is not None:
                            result = cache.get(pick_layer, cell_id)
                            if result is None:
                                fovea = (grid_w / 2.0, grid_h / 2.0)
                                microns_per_px = state.config.retina.microns_per_px
                                if pick_layer == "RGC":
                                    fr = float(state.fr_parasol_on[int(np.clip(grid_y, 0, grid_h - 1)), int(np.clip(grid_x, 0, grid_w - 1))]) if state.fr_parasol_on is not None else 0.0
                                    result = compute_rgc_connectivity(cp, cell_id, fovea, firing_rate=fr)
                                elif pick_layer == "Cone":
                                    result = compute_cone_connectivity(cp, cell_id, fovea)
                                elif pick_layer == "Bipolar":
                                    act = float(state.bp_diffuse_on[int(np.clip(grid_y, 0, grid_h - 1)), int(np.clip(grid_x, 0, grid_w - 1))]) if state.bp_diffuse_on is not None else 0.0
                                    result = compute_bipolar_connectivity(cp, cell_id, activation=act)
                                elif pick_layer == "Horizontal":
                                    result = compute_horizontal_connectivity(cp, cell_id)
                                elif pick_layer == "Amacrine":
                                    result = compute_amacrine_connectivity(cp, cell_id)
                                if result is not None:
                                    cache.put(pick_layer, cell_id, result)
                            update_inspector(result, pick_layer)
                            _shared["picked_cell"] = (pick_layer, cell_id, result)
                            # When in 3D mode, also trigger circuit tracing in Vispy viewer.
                            try:
                                view_mode_click = dpg.get_value("view_mode_combo") if dpg.does_item_exist("view_mode_combo") else "2D Heatmap"
                                if view_mode_click == "3D Stack" and HAS_VISPY and _shared.get("vispy_viewer") is not None:
                                    viewer = _shared.get("vispy_viewer")
                                    if viewer is not None:
                                        viewer.set_selection_from_grid(str(pick_layer), grid_x, grid_y)
                            except Exception:
                                pass
                        else:
                            _shared["picked_cell"] = None
                            update_inspector(None, pick_layer if pick_layer is not None else "RGC")
                except Exception:
                    pass
        _shared["mouse_was_down"] = mouse_down_now
        _shared["mouse_down_pos"] = None if not mouse_down_now else _shared.get("mouse_down_pos")

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

        # Render: 2D heatmap or 3D stack (Vispy)
        view_mode = dpg.get_value("view_mode_combo") if dpg.does_item_exist("view_mode_combo") else "2D Heatmap"
        if view_mode == "3D Stack":
            # Start with a visible fallback frame so we can always see that
            # the 3D mode is active, even if Vispy fails.
            img = np.full((display_h, display_w, 4), [32, 16, 64, 255], dtype=np.uint8)
            if HAS_VISPY:
                # Ensure simulation arrays exist so 3D viewer has something to draw.
                try:
                    state.ensure_initialized()
                except Exception:
                    pass
                try:
                    if _shared.get("vispy_viewer") is None:
                        _shared["vispy_viewer"] = VispyViewer3D(
                            size=(display_w, display_h),
                            config=cfg,
                        )
                    viewer = _shared["vispy_viewer"]
                    if viewer._size != (display_w, display_h):
                        viewer.resize(display_w, display_h)
                    viewer.update_frame(state)
                    # Mouse: drag and scroll zoom (same as before)
                    if dpg.does_item_exist(VIEWPORT_AREA_TAG) and dpg.is_item_hovered(VIEWPORT_AREA_TAG):
                        pos = dpg.get_mouse_pos()
                        if dpg.is_mouse_button_down(0):
                            last = _shared.get("last_mouse_pos")
                            if last is not None:
                                dx, dy = pos[0] - last[0], pos[1] - last[1]
                                viewer.add_drag(dx, dy, sensitivity=0.18)
                            _shared["last_mouse_pos"] = (pos[0], pos[1])
                        else:
                            _shared["last_mouse_pos"] = None
                        wheel = _shared.get("wheel_delta", 0)
                        if wheel != 0:
                            viewer.add_zoom(wheel)
                            _shared["wheel_delta"] = 0
                    else:
                        _shared["last_mouse_pos"] = None
                    vispy_img = viewer.render()
                    if isinstance(vispy_img, np.ndarray) and vispy_img.ndim == 3 and vispy_img.shape[2] >= 3:
                        # Vispy may render at HiDPI resolution (e.g. 2x for Retina),
                        # which can mismatch the Dear PyGui texture size. Downsample
                        # to the dynamic texture resolution so it actually displays.
                        vh, vw = vispy_img.shape[0], vispy_img.shape[1]
                        if vh != display_h or vw != display_w:
                            try:
                                img_pil = Image.fromarray(vispy_img)
                                vispy_img = np.array(
                                    img_pil.resize((display_w, display_h), Image.BILINEAR),
                                    dtype=np.uint8,
                                )
                            except Exception:
                                # Simple stride-based fallback if PIL resize fails
                                step_y = max(1, vh // display_h)
                                step_x = max(1, vw // display_w)
                                vispy_img = vispy_img[::step_y, ::step_x][:display_h, :display_w]
                        img = vispy_img
                except Exception as e:
                    # Minimal stderr logging so we don't crash the UI.
                    print(f"3D viewer error: {e}")
            else:
                # No Vispy: solid fallback color so user still sees 3D mode.
                img = np.full((display_h, display_w, 4), [6, 6, 15, 255], dtype=np.uint8)
            # If the 3D image is effectively all black (e.g. GL failed and returned zeros),
            # fall back to a bright checkerboard so we can visually confirm the 3D path.
            if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] >= 3:
                if int(img.max() if img.size else 0) <= 5:
                    yy, xx = np.indices((display_h, display_w))
                    checker = ((xx // 32) + (yy // 32)) % 2
                    debug_img = np.zeros((display_h, display_w, 4), dtype=np.uint8)
                    debug_img[..., 0] = checker * 255  # red
                    debug_img[..., 1] = (1 - checker) * 255  # green
                    debug_img[..., 2] = 0
                    debug_img[..., 3] = 255
                    img = debug_img
            # Lightweight one-time debug print so we can see what's happening.
            dbg = _shared.get("debug_3d_prints", 0)
            if dbg < 3 and isinstance(img, np.ndarray):
                _shared["debug_3d_prints"] = dbg + 1
                try:
                    print(
                        "3D frame stats:",
                        "shape=", img.shape,
                        "dtype=", img.dtype,
                        "min=", int(img.min() if img.size else 0),
                        "max=", int(img.max() if img.size else 0),
                        "HAS_VISPY=", HAS_VISPY,
                    )
                except Exception:
                    pass
            # 3D returns uint8; DPG wants float 0-1
            tex_data = (img.astype(np.float32) / 255.0).flatten()
        else:
            # 2D heatmap: combo value is display label; resolve to internal key (bio_constants)
            layer_display = dpg.get_value("layer_combo") if dpg.does_item_exist("layer_combo") else LAYER_KEY_TO_DISPLAY.get("RGC Firing (L)", "RGC Firing (L)")
            layer_name = LAYER_DISPLAY_TO_KEY.get(layer_display, layer_display)
            if dpg.does_item_exist("layer_convergence_note"):
                if dpg.does_item_exist("show_convergence_ratios") and dpg.get_value("show_convergence_ratios"):
                    _set_convergence_note(layer_name)
                else:
                    dpg.set_value("layer_convergence_note", "")
            if layer_name == "Stimulus":
                stim_type = state.stimulus_params.get("type", "spot")
                # For image stimuli, show the loaded RGB image directly so the
                # user sees the true pixel colors rather than the spectral
                # centroid approximation used internally for cones.
                if stim_type == "image" and "image_mask" in state.stimulus_params:
                    img = np.asarray(state.stimulus_params["image_mask"], dtype=np.float32)
                    h, w = state.grid_shape()
                    if img.ndim == 2:
                        img = np.stack([img, img, img], axis=-1)
                    if img.shape[0] != h or img.shape[1] != w:
                        img = np.resize(img, (h, w, img.shape[2]))
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
                elif state.stimulus_spectrum is not None:
                    wl = state.config.spectral.wavelengths
                    rgba = spectrum_to_stimulus_rgba(state.stimulus_spectrum, wl)
                else:
                    h, w = state.grid_shape()
                    rgba = np.zeros((h, w, 4), dtype=np.float32)
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
                # Optional: weight by convergence so signal compression is visible (bio_constants)
                if dpg.does_item_exist("biological_scale_2d") and dpg.get_value("biological_scale_2d"):
                    dkey = LAYER_KEY_TO_DENSITY.get(layer_name)
                    if dkey and dkey in RELATIVE_DENSITY:
                        scale = RELATIVE_DENSITY["rgc"] / RELATIVE_DENSITY[dkey]
                        layer = np.clip(layer.astype(np.float32) * scale, 0.0, None)
                rgba = grid_to_rgba(layer, colormap=colormap)
            # Overlay: selected RGC dendritic field and cone/bipolar/amacrine scatter
            picked = _shared.get("picked_cell")
            if picked is not None and len(picked) >= 3:
                _, _, conn_result = picked
                if isinstance(conn_result, RGCConnectivityResult):
                    cp = _shared.get("cell_positions")
                    if cp is not None:
                        overlay = draw_cell_overlay(
                            state.grid_shape(),
                            cp,
                            conn_result,
                            state.config.retina.microns_per_px,
                        )
                        mask = overlay[..., 3:4] > 0
                        rgba = np.where(mask, overlay, rgba)
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

