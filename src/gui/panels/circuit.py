"""View controls, connectivity weights, spike output toggles, and cell parameters."""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from src.gui.app_context import AppContext
from src.gui.panels.stimulus import build_stimulus_section
from src.gui.layout_constants import (
    CONN_F,
    CONN_WEIGHT_MAX,
    CONN_WEIGHT_MIN,
    INPUT_FLOAT_FMT,
    LEFT_PANEL_WIDTH,
    SIDE_COMBO_WIDTH,
    SIDE_SPIN_WIDTH,
)


def build(parent_tag: str, ctx: AppContext) -> str:
    import src.gui.app as app

    state = ctx.sim_state
    lw = int(ctx.shared.get("left_panel_width", LEFT_PANEL_WIDTH))
    root = "panel_circuit_root"
    with dpg.child_window(
        parent=parent_tag,
        tag=root,
        width=lw,
        height=0,
        auto_resize_y=True,
        border=True,
        autosize_x=False,
    ):
        dpg.add_text("Main viewer")
        dpg.add_text(
            "Scroll this column for Spatial heterogeneity below.",
            wrap=max(120, lw - 12),
            color=(140, 160, 180, 255),
        )
        dpg.add_spacer(height=2)
        dpg.add_combo(
            label="Mode",
            items=["2D Heatmap", "2D All Layers"],
            default_value="2D Heatmap",
            tag="view_mode_combo",
            width=SIDE_COMBO_WIDTH,
            callback=lambda s, a: app._update_view_mode_ui(a),
        )
        dpg.add_combo(
            label="Layer",
            items=[label for _k, label in app.LAYER_ITEMS_2D],
            default_value=app.LAYER_KEY_TO_DISPLAY.get("RGC Firing (L)", "RGC Firing (L)"),
            tag="layer_combo",
            width=SIDE_COMBO_WIDTH,
        )
        dpg.add_spacer(height=4)
        with dpg.child_window(
            width=-1,
            height=0,
            auto_resize_y=True,
            border=True,
            autosize_x=False,
            tag="stimulus_section_wrap",
        ):
            build_stimulus_section(state, app)
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
            width=SIDE_COMBO_WIDTH,
        )
        app._panel_section_gap()
        dpg.add_text("Pathway weights")
        dpg.add_text("Connectivity gains (-3 to 3).")
        dpg.add_text("Negative values invert / scale inhibitory paths.")
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
            def _conn_cb(sender, app_data, _k=key):
                app._set_conn_weight(state, _k, app_data)
                app._set_connectivity_dirty()

            dpg.add_input_float(
                label=label,
                default_value=getattr(cw, key),
                min_value=CONN_WEIGHT_MIN,
                max_value=CONN_WEIGHT_MAX,
                min_clamped=True,
                max_clamped=True,
                width=SIDE_SPIN_WIDTH,
                tag=tag,
                **CONN_F,
                callback=_conn_cb,
            )
        dpg.add_button(
            label="Reset weights to 1.0",
            tag="conn_reset",
            width=-1,
            callback=lambda: app._reset_connectivity_weights(state),
        )
        dpg.add_button(
            label="Randomize weights",
            tag="conn_randomize",
            width=-1,
            callback=lambda: app._randomize_connectivity_weights(state),
        )
        app._panel_section_gap()
        with dpg.tree_node(label="Spike output (Poisson)", default_open=False):
            dpg.add_checkbox(
                label="Generate spikes from RGC rates (Poisson)",
                default_value=state.config.spike_output.enabled,
                tag="spike_output_enabled",
                callback=lambda s, a: (
                    setattr(state.config.spike_output, "enabled", bool(a)),
                    app._toggle_spike_raster_ui(bool(a)),
                ),
            )
            dpg.add_checkbox(
                label="Spikes from smoothed rates (else LN instant)",
                default_value=state.config.spike_output.use_smoothed_rates,
                tag="spike_use_smoothed",
                callback=lambda s, a: setattr(state.config.spike_output, "use_smoothed_rates", bool(a)),
            )
        with dpg.tree_node(label="Detailed neuron / pathway parameters", default_open=False):
            cfg = state.config
            with dpg.tree_node(label="RGC pathway (narrow field)", default_open=True):
                dpg.add_input_float(
                    label="Dendritic sigma (deg)",
                    default_value=cfg.dendritic.sigma_midget_deg,
                    step=0.001,
                    **INPUT_FLOAT_FMT,
                    width=SIDE_SPIN_WIDTH,
                    callback=lambda s, a: setattr(cfg.dendritic, "sigma_midget_deg", a),
                )
                dpg.add_input_float(
                    label="Max firing (sp/s)",
                    default_value=cfg.rgc_nl.r_max,
                    step=1.0,
                    **INPUT_FLOAT_FMT,
                    width=SIDE_SPIN_WIDTH,
                    callback=lambda s, a: setattr(cfg.rgc_nl, "r_max", a),
                )
                dpg.add_input_float(
                    label="LN slope",
                    default_value=cfg.rgc_nl.slope,
                    step=0.1,
                    **INPUT_FLOAT_FMT,
                    width=SIDE_SPIN_WIDTH,
                    callback=lambda s, a: setattr(cfg.rgc_nl, "slope", a),
                )
                dpg.add_input_float(
                    label="LN half-point",
                    default_value=cfg.rgc_nl.x_half,
                    step=0.01,
                    **INPUT_FLOAT_FMT,
                    width=SIDE_SPIN_WIDTH,
                    callback=lambda s, a: setattr(cfg.rgc_nl, "x_half", a),
                )
                dpg.add_input_float(
                    label="Tau (s)",
                    default_value=cfg.temporal.rgc_tau,
                    step=0.01,
                    **INPUT_FLOAT_FMT,
                    width=SIDE_SPIN_WIDTH,
                    callback=lambda s, a: setattr(cfg.temporal, "rgc_tau", a),
                )
            with dpg.tree_node(label="RGC pathway (wide field)", default_open=False):
                dpg.add_input_float(
                    label="Dendritic sigma (deg)",
                    default_value=cfg.dendritic.sigma_parasol_deg,
                    step=0.001,
                    **INPUT_FLOAT_FMT,
                    width=SIDE_SPIN_WIDTH,
                    callback=lambda s, a: setattr(cfg.dendritic, "sigma_parasol_deg", a),
                )
                dpg.add_text("(LN r_max / slope / half-point shared with narrow field above)")
            with dpg.tree_node(label="Bipolar pooling", default_open=False):
                dpg.add_input_float(
                    label="Sigma diffuse (deg)",
                    default_value=cfg.bipolar.sigma_diffuse_deg,
                    step=0.001,
                    **INPUT_FLOAT_FMT,
                    width=SIDE_SPIN_WIDTH,
                    callback=lambda s, a: setattr(cfg.bipolar, "sigma_diffuse_deg", a),
                )
                dpg.add_input_float(
                    label="Tau (s)",
                    default_value=cfg.temporal.bipolar_tau,
                    step=0.01,
                    **INPUT_FLOAT_FMT,
                    width=SIDE_SPIN_WIDTH,
                    callback=lambda s, a: setattr(cfg.temporal, "bipolar_tau", a),
                )
            with dpg.tree_node(label="Horizontal feedback", default_open=False):
                dpg.add_input_float(
                    label="Sigma LM (deg)",
                    default_value=cfg.horizontal.sigma_lm_deg,
                    step=0.001,
                    **INPUT_FLOAT_FMT,
                    width=SIDE_SPIN_WIDTH,
                    callback=lambda s, a: setattr(cfg.horizontal, "sigma_lm_deg", a),
                )
                dpg.add_input_float(
                    label="Sigma S (deg)",
                    default_value=cfg.horizontal.sigma_s_deg,
                    step=0.001,
                    **INPUT_FLOAT_FMT,
                    width=SIDE_SPIN_WIDTH,
                    callback=lambda s, a: setattr(cfg.horizontal, "sigma_s_deg", a),
                )
                dpg.add_input_float(
                    label="Alpha LM",
                    default_value=cfg.horizontal.alpha_lm,
                    step=0.01,
                    **INPUT_FLOAT_FMT,
                    width=SIDE_SPIN_WIDTH,
                    callback=lambda s, a: setattr(cfg.horizontal, "alpha_lm", a),
                )
                dpg.add_input_float(
                    label="Alpha S",
                    default_value=cfg.horizontal.alpha_s,
                    step=0.01,
                    **INPUT_FLOAT_FMT,
                    width=SIDE_SPIN_WIDTH,
                    callback=lambda s, a: setattr(cfg.horizontal, "alpha_s", a),
                )
                dpg.add_input_float(
                    label="Tau (s)",
                    default_value=cfg.temporal.horizontal_tau,
                    step=0.01,
                    **INPUT_FLOAT_FMT,
                    width=SIDE_SPIN_WIDTH,
                    callback=lambda s, a: setattr(cfg.temporal, "horizontal_tau", a),
                )
            with dpg.tree_node(label="Lateral inhibition (narrow pool)", default_open=False):
                dpg.add_input_float(
                    label="Sigma (deg)",
                    default_value=cfg.amacrine.sigma_aii_deg,
                    step=0.001,
                    **INPUT_FLOAT_FMT,
                    width=SIDE_SPIN_WIDTH,
                    callback=lambda s, a: setattr(cfg.amacrine, "sigma_aii_deg", a),
                )
                dpg.add_input_float(
                    label="Gamma (weight)",
                    default_value=cfg.amacrine.gamma_aii,
                    step=0.01,
                    **INPUT_FLOAT_FMT,
                    width=SIDE_SPIN_WIDTH,
                    callback=lambda s, a: setattr(cfg.amacrine, "gamma_aii", a),
                )
                dpg.add_input_float(
                    label="Tau (s)",
                    default_value=cfg.temporal.amacrine_tau,
                    step=0.01,
                    **INPUT_FLOAT_FMT,
                    width=SIDE_SPIN_WIDTH,
                    callback=lambda s, a: setattr(cfg.temporal, "amacrine_tau", a),
                )
            with dpg.tree_node(label="Lateral inhibition (wide pool)", default_open=False):
                dpg.add_input_float(
                    label="Sigma (deg)",
                    default_value=cfg.amacrine.sigma_wide_deg,
                    step=0.001,
                    **INPUT_FLOAT_FMT,
                    width=SIDE_SPIN_WIDTH,
                    callback=lambda s, a: setattr(cfg.amacrine, "sigma_wide_deg", a),
                )
                dpg.add_input_float(
                    label="Gamma (weight)",
                    default_value=cfg.amacrine.gamma_wide,
                    step=0.01,
                    **INPUT_FLOAT_FMT,
                    width=SIDE_SPIN_WIDTH,
                    callback=lambda s, a: setattr(cfg.amacrine, "gamma_wide", a),
                )
                dpg.add_text("(Tau shared with narrow pool above)")
                dpg.add_spacer(height=8)
    return root
