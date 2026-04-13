"""Spatial heterogeneity controls panel."""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from src.config import EccentricityGradientType, MosaicLayoutType, SpatialHeterogeneityMode
from src.gui.app_context import AppContext
from src.gui.layout_constants import (
    CONN_F,
    LEFT_PANEL_WIDTH,
    SIDE_COMBO_WIDTH,
    SIDE_PANEL_TEXT_WRAP,
    SIDE_SLIDER_WIDTH,
    SIDE_SPIN_WIDE,
    SIDE_SPIN_WIDTH,
)
from src.simulation import SimState
from src.simulation.bio_constants import EMPIRICAL_ECCENTRICITY_AVAILABLE

SH_MODE_LABELS = (
    "Homogeneous",
    "Parameter scatter",
    "Discrete type map",
    "Eccentricity gradient",
    "Voronoi mosaic",
)

SH_TYPE_BUCKET_LABELS = (
    "Midget ON",
    "Midget OFF",
    "Parasol ON",
    "Parasol OFF",
    "Bistratified",
    "Other",
)


def build(parent_tag: str, ctx: AppContext) -> str:
    import src.gui.app as app

    root = "panel_heterogeneity_root"
    lw = int(ctx.shared.get("left_panel_width", LEFT_PANEL_WIDTH))
    with dpg.child_window(
        parent=parent_tag,
        tag=root,
        width=lw,
        height=0,
        auto_resize_y=True,
        border=True,
        autosize_x=False,
    ):
        # Do not wrap this panel in tree_node/collapsing_header: DPG can treat
        # off-screen child_window regions as "skipped" and persist a false
        # collapsed state, hiding all mode controls (hoffstadt/DearPyGui#1873).
        _build_heterogeneity_ui(ctx.sim_state, app)
    return root


def _build_heterogeneity_ui(state: SimState, app) -> None:
    cfg = state.config
    sh = cfg.spatial_heterogeneity
    desc_color = (160, 200, 220, 255)

    def on_mode_change(s, app_data) -> None:
        try:
            idx = SH_MODE_LABELS.index(str(app_data))
        except ValueError:
            return
        modes = (
            SpatialHeterogeneityMode.HOMOGENEOUS,
            SpatialHeterogeneityMode.SCATTER,
            SpatialHeterogeneityMode.TYPE_MAP,
            SpatialHeterogeneityMode.ECCENTRICITY,
            SpatialHeterogeneityMode.MOSAIC,
        )
        sh.mode = modes[idx]
        app._sh_sync_mode_group_visibility(idx)
        app._sh_mark_heterogeneity_dirty(state)

    dpg.add_text("Spatial heterogeneity")
    dpg.add_radio_button(
        items=list(SH_MODE_LABELS),
        default_value=SH_MODE_LABELS[app._sh_mode_index(sh.mode)],
        horizontal=False,
        tag="sh_mode_radio",
        callback=on_mode_change,
    )

    # --- Mode 0 ---
    with dpg.group(tag="sh_mode_group_0", show=sh.mode == SpatialHeterogeneityMode.HOMOGENEOUS):
        dpg.add_spacer(height=4)
        dpg.add_text(
            "All cells share identical parameters.",
            color=desc_color,
            wrap=SIDE_PANEL_TEXT_WRAP,
        )
        dpg.add_text(
            "Mean-field approximation.",
            color=desc_color,
            wrap=SIDE_PANEL_TEXT_WRAP,
        )

    # --- Mode 1 ---
    sc = sh.scatter
    with dpg.group(tag="sh_mode_group_1", show=sh.mode == SpatialHeterogeneityMode.SCATTER):
        dpg.add_spacer(height=4)
        dpg.add_text(
            "Biological variability: each cell's weights are drawn from a distribution around the mean, mimicking cell-to-cell noise in real tissue.",
            color=desc_color,
            wrap=SIDE_PANEL_TEXT_WRAP,
        )
        dpg.add_slider_float(
            label="Scatter sigma",
            min_value=0.0,
            max_value=1.0,
            default_value=float(sc.sigma),
            width=SIDE_SLIDER_WIDTH,
            tag="sh_sc_sigma",
            callback=lambda s, a: (
                setattr(sc, "sigma", float(a)),
                app._sh_mark_heterogeneity_dirty(state),
            ),
        )
        dpg.add_checkbox(
            label="cone_to_bipolar",
            default_value=sc.affect_cone_to_bipolar,
            tag="sh_sc_cb",
            callback=lambda s, a: (
                setattr(sc, "affect_cone_to_bipolar", bool(a)),
                app._sh_mark_heterogeneity_dirty(state),
            ),
        )
        dpg.add_checkbox(
            label="bipolar_to_rgc",
            default_value=sc.affect_bipolar_to_rgc,
            tag="sh_sc_br",
            callback=lambda s, a: (
                setattr(sc, "affect_bipolar_to_rgc", bool(a)),
                app._sh_mark_heterogeneity_dirty(state),
            ),
        )
        dpg.add_checkbox(
            label="amacrine_to_bipolar",
            default_value=sc.affect_amacrine_to_bipolar,
            tag="sh_sc_ab",
            callback=lambda s, a: (
                setattr(sc, "affect_amacrine_to_bipolar", bool(a)),
                app._sh_mark_heterogeneity_dirty(state),
            ),
        )
        dpg.add_button(
            label="Resample noise map",
            width=-1,
            callback=lambda: (
                setattr(sc, "resample_seed", int(sc.resample_seed) + 1),
                app._sh_mark_heterogeneity_dirty(state),
            ),
        )

    # --- Mode 2 ---
    tm = sh.type_map
    with dpg.group(tag="sh_mode_group_2", show=sh.mode == SpatialHeterogeneityMode.TYPE_MAP):
        dpg.add_spacer(height=4)
        dpg.add_text(
            "Each spatial location is assigned an RGC type identity drawn from population fractions. RF size, gain, and temporal tuning vary by type.",
            color=desc_color,
            wrap=SIDE_PANEL_TEXT_WRAP,
        )
        for i in range(6):
            dpg.add_slider_float(
                label=SH_TYPE_BUCKET_LABELS[i],
                min_value=0.0,
                max_value=1.0,
                default_value=float(tm.type_fractions[i]),
                width=SIDE_SLIDER_WIDTH,
                tag=f"sh_tm_frac_{i}",
                callback=lambda s, a, ii=i: (
                    app._sh_normalize_tm_fractions(state),
                    app._sh_mark_heterogeneity_dirty(state),
                ),
            )
        dpg.add_text("Type fractions (%)", tag="sh_tm_frac_readout")
        dpg.add_separator()
        dpg.add_text("RF size multiplier")
        for i in range(6):
            dpg.add_input_float(
                label=SH_TYPE_BUCKET_LABELS[i],
                default_value=float(tm.rf_multiplier[i]),
                min_value=0.1,
                max_value=4.0,
                min_clamped=True,
                max_clamped=True,
                width=SIDE_SPIN_WIDTH,
                tag=f"sh_tm_rf_{i}",
                **CONN_F,
                callback=lambda s, a, ii=i: (
                    app._tm_set_rf(state, ii, float(a)),
                    app._sh_mark_heterogeneity_dirty(state),
                ),
            )
        dpg.add_text("Gain multiplier")
        for i in range(6):
            dpg.add_input_float(
                label=SH_TYPE_BUCKET_LABELS[i],
                default_value=float(tm.gain_multiplier[i]),
                min_value=0.1,
                max_value=4.0,
                min_clamped=True,
                max_clamped=True,
                width=SIDE_SPIN_WIDTH,
                tag=f"sh_tm_gn_{i}",
                **CONN_F,
                callback=lambda s, a, ii=i: (
                    app._tm_set_gain(state, ii, float(a)),
                    app._sh_mark_heterogeneity_dirty(state),
                ),
            )
        dpg.add_button(
            label="Regenerate type map",
            width=-1,
            callback=lambda: (
                setattr(tm, "map_seed", int(tm.map_seed) + 1),
                app._sh_mark_heterogeneity_dirty(state),
            ),
        )

    # --- Mode 3 ---
    ec = sh.eccentricity
    ecc_items = ["Linear", "Square root"]
    if EMPIRICAL_ECCENTRICITY_AVAILABLE:
        ecc_items.append("Empirical (Curcio and Allen)")
    with dpg.group(tag="sh_mode_group_3", show=sh.mode == SpatialHeterogeneityMode.ECCENTRICITY):
        dpg.add_spacer(height=4)
        dpg.add_text(
            "RF size grows and cell density decreases with distance from the foveal center, matching known human retinal topography.",
            color=desc_color,
            wrap=SIDE_PANEL_TEXT_WRAP,
        )
        dpg.add_slider_float(
            label="Fovea X (px)",
            min_value=0.0,
            max_value=float(cfg.retina.grid_resolution - 1),
            default_value=float(ec.fovea_px_x),
            width=SIDE_SLIDER_WIDTH,
            tag="sh_ec_fx",
            callback=lambda s, a: (
                setattr(ec, "fovea_px_x", float(a)),
                app._sh_mark_heterogeneity_dirty(state),
            ),
        )
        dpg.add_slider_float(
            label="Fovea Y (px)",
            min_value=0.0,
            max_value=float(cfg.retina.grid_resolution - 1),
            default_value=float(ec.fovea_px_y),
            width=SIDE_SLIDER_WIDTH,
            tag="sh_ec_fy",
            callback=lambda s, a: (
                setattr(ec, "fovea_px_y", float(a)),
                app._sh_mark_heterogeneity_dirty(state),
            ),
        )
        dpg.add_input_float(
            label="Eccentricity scale (deg/px)",
            default_value=float(ec.eccentricity_scale_deg_per_px),
            min_value=0.001,
            max_value=0.5,
            min_clamped=True,
            max_clamped=True,
            width=SIDE_SPIN_WIDE,
            tag="sh_ec_scale",
            **CONN_F,
            callback=lambda s, a: (
                setattr(ec, "eccentricity_scale_deg_per_px", float(a)),
                app._sh_mark_heterogeneity_dirty(state),
            ),
        )
        dpg.add_input_float(
            label="RF growth strength",
            default_value=float(ec.rf_growth_strength),
            min_value=0.0,
            max_value=3.0,
            min_clamped=True,
            max_clamped=True,
            width=SIDE_SPIN_WIDE,
            tag="sh_ec_strength",
            **CONN_F,
            callback=lambda s, a: (
                setattr(ec, "rf_growth_strength", float(a)),
                app._sh_mark_heterogeneity_dirty(state),
            ),
        )

        def _ec_combo_to_enum(label: str) -> EccentricityGradientType:
            if label.startswith("Linear"):
                return EccentricityGradientType.LINEAR
            if label.startswith("Square"):
                return EccentricityGradientType.SQRT
            return EccentricityGradientType.EMPIRICAL

        def _ec_enum_to_combo(e: EccentricityGradientType) -> str:
            if e == EccentricityGradientType.LINEAR:
                return ecc_items[0]
            if e == EccentricityGradientType.SQRT:
                return ecc_items[1]
            return ecc_items[-1] if len(ecc_items) > 2 else ecc_items[0]

        dpg.add_combo(
            label="Gradient function",
            items=ecc_items,
            default_value=_ec_enum_to_combo(ec.gradient),
            width=SIDE_COMBO_WIDTH,
            tag="sh_ec_grad",
            callback=lambda s, a: (
                setattr(ec, "gradient", _ec_combo_to_enum(str(a))),
                app._sh_mark_heterogeneity_dirty(state),
            ),
        )
        dpg.add_checkbox(
            label="Preview eccentricity isolines on 2D heatmap",
            default_value=ec.preview_overlay,
            tag="sh_ec_preview",
            callback=lambda s, a: setattr(ec, "preview_overlay", bool(a)),
        )

    # --- Mode 4 ---
    mo = sh.mosaic
    mosaic_labels = ("Hexagonal regular", "Hexagonal + jitter", "Random Poisson")

    def _mos_combo_to_enum(lab: str) -> MosaicLayoutType:
        if "regular" in lab.lower():
            return MosaicLayoutType.HEX_REGULAR
        if "jitter" in lab.lower():
            return MosaicLayoutType.HEX_JITTER
        return MosaicLayoutType.POISSON

    def _mos_enum_to_combo(m: MosaicLayoutType) -> str:
        if m == MosaicLayoutType.HEX_REGULAR:
            return mosaic_labels[0]
        if m == MosaicLayoutType.HEX_JITTER:
            return mosaic_labels[1]
        return mosaic_labels[2]

    def on_mosaic_type(s, app_data) -> None:
        mo.mosaic_type = _mos_combo_to_enum(str(app_data))
        show_j = mo.mosaic_type == MosaicLayoutType.HEX_JITTER
        if dpg.does_item_exist("sh_mosaic_jitter_group"):
            dpg.configure_item("sh_mosaic_jitter_group", show=show_j)
        app._sh_mark_heterogeneity_dirty(state)

    with dpg.group(tag="sh_mode_group_4", show=sh.mode == SpatialHeterogeneityMode.MOSAIC):
        dpg.add_spacer(height=4)
        dpg.add_text(
            "A discrete set of RGC units tile the retina; each unit integrates inputs from its Voronoi territory.",
            color=desc_color,
            wrap=SIDE_PANEL_TEXT_WRAP,
        )
        def on_mo_n_cells(s, app_data) -> None:
            v = int(app_data)
            mo.n_cells = v
            if dpg.does_item_exist("sh_mo_warn_group"):
                dpg.configure_item("sh_mo_warn_group", show=v > 500)
            app._sh_mark_heterogeneity_dirty(state)

        dpg.add_slider_int(
            label="N cells",
            min_value=100,
            max_value=2000,
            default_value=int(mo.n_cells),
            width=SIDE_SLIDER_WIDTH,
            tag="sh_mo_n",
            callback=on_mo_n_cells,
        )
        with dpg.group(tag="sh_mo_warn_group", show=int(mo.n_cells) > 500):
            dpg.add_text(
                "Large N can slow the simulation. Consider N <= 500 for interactive use.",
                color=(220, 180, 120, 255),
            )
        dpg.add_combo(
            label="Mosaic type",
            items=list(mosaic_labels),
            default_value=_mos_enum_to_combo(mo.mosaic_type),
            width=SIDE_COMBO_WIDTH,
            tag="sh_mo_type",
            callback=on_mosaic_type,
        )
        with dpg.group(
            tag="sh_mosaic_jitter_group",
            show=mo.mosaic_type == MosaicLayoutType.HEX_JITTER,
        ):
            dpg.add_slider_float(
                label="Jitter sigma",
                min_value=0.0,
                max_value=1.0,
                default_value=float(mo.jitter_sigma),
                width=SIDE_SLIDER_WIDTH,
                tag="sh_mo_jitter",
                callback=lambda s, a: (
                    setattr(mo, "jitter_sigma", float(a)),
                    app._sh_mark_heterogeneity_dirty(state),
                ),
            )
        dpg.add_checkbox(
            label="Show mosaic overlay (Voronoi boundaries + centers)",
            default_value=mo.show_overlay,
            tag="sh_mo_overlay",
            callback=lambda s, a: setattr(mo, "show_overlay", bool(a)),
        )
        dpg.add_checkbox(
            label="Static snapshot while rebuilding mosaic (pauses sim one frame)",
            default_value=mo.static_snapshot_while_building,
            tag="sh_mo_static",
            callback=lambda s, a: setattr(
                mo, "static_snapshot_while_building", bool(a)
            ),
        )
        dpg.add_button(
            label="Regenerate mosaic",
            width=-1,
            callback=lambda: (
                setattr(mo, "mosaic_seed", int(mo.mosaic_seed) + 1),
                app._sh_mark_heterogeneity_dirty(state),
            ),
        )

    app._sh_sync_mode_group_visibility(app._sh_mode_index(sh.mode))
    if dpg.does_item_exist("sh_tm_frac_0"):
        app._sh_normalize_tm_fractions(state)
