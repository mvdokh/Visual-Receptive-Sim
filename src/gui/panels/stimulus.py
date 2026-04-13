"""Stimulus parameters (embedded in the main viewer / circuit panel)."""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from src.gui.layout_constants import STIM_COMBO_WIDTH, STIM_SLIDER_WIDTH
from src.simulation import SimState


def build_stimulus_section(state: SimState, app) -> None:
    """Add stimulus widgets to the current Dear PyGui parent (no outer child_window)."""
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
        width=STIM_COMBO_WIDTH,
        callback=lambda s, a: app.apply_stimulus_type_change_from_ui(str(a)),
    )
    dpg.add_slider_float(
        label="Wavelength (nm)",
        min_value=380,
        max_value=700,
        default_value=550,
        width=STIM_SLIDER_WIDTH,
        callback=lambda s, a: state.stimulus_params.update({"wavelength_nm": a}),
    )
    dpg.add_slider_float(
        label="Intensity",
        min_value=0.0,
        max_value=1.0,
        default_value=1.0,
        width=STIM_SLIDER_WIDTH,
        callback=lambda s, a: state.stimulus_params.update({"intensity": a}),
    )
    dpg.add_slider_float(
        label="Radius (deg)",
        min_value=0.02,
        max_value=0.5,
        default_value=0.15,
        width=STIM_SLIDER_WIDTH,
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
        dpg.add_slider_float(
            label="X center (deg)",
            min_value=-0.5,
            max_value=0.5,
            default_value=0.0,
            width=STIM_SLIDER_WIDTH,
            tag="stim_x_deg",
            callback=lambda s, a: state.stimulus_params.update({"x_deg": a}),
        )
        dpg.add_slider_float(
            label="Y center (deg)",
            min_value=-0.5,
            max_value=0.5,
            default_value=0.0,
            width=STIM_SLIDER_WIDTH,
            tag="stim_y_deg",
            callback=lambda s, a: state.stimulus_params.update({"y_deg": a}),
        )
        dpg.add_slider_float(
            label="Orientation (deg)",
            min_value=0.0,
            max_value=180.0,
            default_value=0.0,
            width=STIM_SLIDER_WIDTH,
            tag="stim_orientation",
            callback=lambda s, a: state.stimulus_params.update({"orientation_deg": a}),
        )
        dpg.add_slider_float(
            label="Width (deg)",
            min_value=0.02,
            max_value=0.4,
            default_value=0.1,
            width=STIM_SLIDER_WIDTH,
            tag="stim_width",
            callback=lambda s, a: state.stimulus_params.update({"width_deg": a}),
        )
        dpg.add_slider_float(
            label="Spatial freq (cpd)",
            min_value=0.5,
            max_value=8.0,
            default_value=2.0,
            width=STIM_SLIDER_WIDTH,
            tag="stim_spatial_freq",
            callback=lambda s, a: state.stimulus_params.update({"spatial_freq_cpd": a}),
        )
        dpg.add_slider_float(
            label="Phase (deg)",
            min_value=0.0,
            max_value=360.0,
            default_value=0.0,
            width=STIM_SLIDER_WIDTH,
            tag="stim_phase",
            callback=lambda s, a: state.stimulus_params.update({"phase_deg": a}),
        )
        dpg.add_slider_float(
            label="Inner radius (deg)",
            min_value=0.01,
            max_value=0.3,
            default_value=0.05,
            width=STIM_SLIDER_WIDTH,
            tag="stim_inner_radius",
            callback=lambda s, a: state.stimulus_params.update({"inner_radius_deg": a}),
        )
        dpg.add_slider_float(
            label="Velocity X (deg/s)",
            min_value=-2.0,
            max_value=2.0,
            default_value=0.0,
            width=STIM_SLIDER_WIDTH,
            tag="stim_vx",
            callback=lambda s, a: state.stimulus_params.update({"vx_deg_s": a}),
        )
        dpg.add_slider_float(
            label="Velocity Y (deg/s)",
            min_value=-2.0,
            max_value=2.0,
            default_value=0.0,
            width=STIM_SLIDER_WIDTH,
            tag="stim_vy",
            callback=lambda s, a: state.stimulus_params.update({"vy_deg_s": a}),
        )
        dpg.add_combo(
            label="Motion mode",
            items=["linear", "loop", "oscillate"],
            default_value="linear",
            tag="stim_motion_mode",
            width=STIM_COMBO_WIDTH,
            callback=lambda s, a: state.stimulus_params.update({"motion_mode": a}),
        )
        dpg.add_slider_float(
            label="Loop period (s)",
            min_value=0.2,
            max_value=20.0,
            default_value=2.0,
            width=STIM_SLIDER_WIDTH,
            tag="stim_motion_period",
            callback=lambda s, a: state.stimulus_params.update({"motion_period_s": a}),
        )
        dpg.add_slider_float(
            label="Oscillate amplitude (deg)",
            min_value=0.02,
            max_value=0.5,
            default_value=0.2,
            width=STIM_SLIDER_WIDTH,
            tag="stim_motion_amp",
            callback=lambda s, a: state.stimulus_params.update({"motion_osc_amp_deg": a}),
        )
        dpg.add_slider_float(
            label="Oscillate frequency (Hz)",
            min_value=0.05,
            max_value=5.0,
            default_value=1.0,
            width=STIM_SLIDER_WIDTH,
            tag="stim_motion_hz",
            callback=lambda s, a: state.stimulus_params.update({"motion_osc_hz": a}),
        )
        dpg.add_slider_float(
            label="Secondary radius (deg)",
            min_value=0.02,
            max_value=0.5,
            default_value=0.15,
            width=STIM_SLIDER_WIDTH,
            tag="stim_radius2",
            callback=lambda s, a: state.stimulus_params.update({"radius2_deg": a}),
        )
        dpg.add_slider_float(
            label="Secondary X (deg)",
            min_value=-0.5,
            max_value=0.5,
            default_value=0.25,
            width=STIM_SLIDER_WIDTH,
            tag="stim_x2_deg",
            callback=lambda s, a: state.stimulus_params.update({"x2_deg": a}),
        )
        dpg.add_slider_float(
            label="Secondary Y (deg)",
            min_value=-0.5,
            max_value=0.5,
            default_value=0.0,
            width=STIM_SLIDER_WIDTH,
            tag="stim_y2_deg",
            callback=lambda s, a: state.stimulus_params.update({"y2_deg": a}),
        )
        dpg.add_slider_float(
            label="Secondary wavelength (nm)",
            min_value=380,
            max_value=700,
            default_value=450,
            width=STIM_SLIDER_WIDTH,
            tag="stim_wavelength2",
            callback=lambda s, a: state.stimulus_params.update({"wavelength2_nm": a}),
        )
        dpg.add_slider_float(
            label="Secondary intensity",
            min_value=0.0,
            max_value=1.0,
            default_value=1.0,
            width=STIM_SLIDER_WIDTH,
            tag="stim_intensity2",
            callback=lambda s, a: state.stimulus_params.update({"intensity2": a}),
        )
