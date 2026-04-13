"""Right column: tabbed stats, dynamics, cone/opponent plots, raster, export & recording."""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from src.gui.app_context import AppContext
from src.gui.layout_constants import (
    RIGHT_PANEL_WIDTH,
    SIDE_PANEL_TEXT_WRAP,
    SIDE_SLIDER_WIDTH,
    SIDE_SPIN_WIDTH,
)

# Order matches ``layer_data`` keys in ``app._update_stats`` (single source for UI + updates).
STAT_LAYER_PLOT_ORDER = (
    "Stimulus",
    "Cones L",
    "Cones M",
    "Cones S",
    "Horizontal",
    "Bipolar",
    "Amacrine",
    "RGC",
)


def _stat_layer_slug(name: str) -> str:
    return name.replace(" ", "_")


# Bar fill colors for per-layer mean / std / min / max (shared legend in Stats tab).
STAT_METRIC_SERIES_COLORS_RGBA = {
    "mean": (88, 156, 255, 255),
    "std": (255, 178, 76, 255),
    "min": (96, 210, 145, 255),
    "max": (210, 98, 255, 255),
}


def _ensure_stat_metric_bar_themes() -> None:
    if dpg.does_item_exist("stat_metric_theme_mean"):
        return
    for key, rgba in STAT_METRIC_SERIES_COLORS_RGBA.items():
        with dpg.theme(tag=f"stat_metric_theme_{key}"):
            with dpg.theme_component(dpg.mvBarSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Fill, rgba)


def build(parent_tag: str, ctx: AppContext) -> str:
    import src.gui.app as app

    state = ctx.sim_state
    rw = int(ctx.shared.get("right_panel_width", RIGHT_PANEL_WIDTH))
    root = "panel_stats_plots_root"
    wrap = min(SIDE_PANEL_TEXT_WRAP, rw - 16)
    _scale_cb = lambda s, a: app._sync_right_panel_plot_axes()

    with dpg.child_window(
        parent=parent_tag,
        tag=root,
        width=rw,
        height=-1,
        border=True,
        autosize_x=False,
    ):
        with dpg.tab_bar(tag="right_panel_tabs"):
            _ensure_stat_metric_bar_themes()

            with dpg.tab(label="Dynamics"):
                dpg.add_text(
                    "RGC mean firing rate — 10 samples/s, last 10 s (100 points)",
                    tag="sparkline_label",
                )
                dpg.add_checkbox(
                    label="Log10 Y (firing rate)",
                    tag="plot_log_spark_y",
                    default_value=False,
                    callback=_scale_cb,
                )
                with dpg.plot(tag="sparkline_plot", height=160, width=-1, anti_aliased=True):
                    dpg.add_plot_axis(
                        dpg.mvXAxis,
                        tag="spark_x",
                        label="s",
                        no_gridlines=False,
                        foreground_grid=True,
                    )
                    dpg.add_plot_axis(
                        dpg.mvYAxis,
                        tag="spark_y",
                        no_gridlines=False,
                        foreground_grid=True,
                    )
                    dpg.add_line_series([], [], tag="sparkline_series", parent="spark_y")
                dpg.add_spacer(height=6)
                dpg.add_text(
                    "RGC firing-rate histogram (grid) — updates at 10 Hz",
                    tag="hist_label",
                )
                dpg.add_checkbox(
                    label="Log10 X (rate bins)",
                    tag="plot_log_hist_x",
                    default_value=False,
                    callback=_scale_cb,
                )
                dpg.add_checkbox(
                    label="Log10 Y (counts)",
                    tag="plot_log_hist_y",
                    default_value=False,
                    callback=_scale_cb,
                )
                with dpg.plot(tag="hist_plot", height=160, width=-1, anti_aliased=True):
                    dpg.add_plot_axis(
                        dpg.mvXAxis,
                        tag="hist_x",
                        no_gridlines=False,
                        foreground_grid=True,
                    )
                    dpg.add_plot_axis(
                        dpg.mvYAxis,
                        tag="hist_y",
                        no_gridlines=False,
                        foreground_grid=True,
                    )
                    dpg.add_bar_series([], [], tag="hist_series", parent="hist_y", weight=0.8)

            with dpg.tab(label="Stats"):
                dpg.add_text("Opponent summaries")
                dpg.add_text("", tag="lm_summary")
                dpg.add_text("", tag="by_summary")
                dpg.add_separator()
                dpg.add_text("Per-layer grid statistics (mean, std, min, max over space)")
                with dpg.group(horizontal=True):
                    dpg.add_text("Colors:", color=(150, 168, 188, 255))
                    for label, key in (
                        ("mean", "mean"),
                        ("std", "std"),
                        ("min", "min"),
                        ("max", "max"),
                    ):
                        dpg.add_spacer(width=10)
                        dpg.add_text(label, color=STAT_METRIC_SERIES_COLORS_RGBA[key])
                dpg.add_spacer(height=6)
                for name in STAT_LAYER_PLOT_ORDER:
                    slug = _stat_layer_slug(name)
                    dpg.add_text(name, color=(180, 200, 220, 255))
                    ax_tag = f"stat_layer_ax_{slug}"
                    ay_tag = f"stat_layer_ay_{slug}"
                    with dpg.plot(
                        tag=f"stat_layer_plot_{slug}",
                        height=82,
                        width=-1,
                        anti_aliased=True,
                    ):
                        dpg.add_plot_axis(
                            dpg.mvXAxis,
                            tag=ax_tag,
                            no_gridlines=False,
                        )
                        dpg.add_plot_axis(
                            dpg.mvYAxis,
                            tag=ay_tag,
                            no_gridlines=False,
                        )
                        dpg.set_axis_ticks(
                            ax_tag,
                            (("mean", 0.0), ("std", 1.0), ("min", 2.0), ("max", 3.0)),
                        )
                        for i, metric in enumerate(("mean", "std", "min", "max")):
                            ser_tag = f"stat_layer_series_{slug}_{metric}"
                            dpg.add_bar_series(
                                [float(i)],
                                [0.0],
                                weight=0.82,
                                parent=ay_tag,
                                tag=ser_tag,
                            )
                            dpg.bind_item_theme(ser_tag, f"stat_metric_theme_{metric}")

            with dpg.tab(label="Cone & opponent"):
                dpg.add_text("Cone mean drive (L / M / S)")
                dpg.add_checkbox(
                    label="Log10 Y (positive cone means only)",
                    tag="plot_log_cone_y",
                    default_value=False,
                    callback=_scale_cb,
                )
                with dpg.plot(height=180, width=-1, tag="plot_cone_act", anti_aliased=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(
                        dpg.mvXAxis,
                        label="",
                        tag="plot_cone_ax",
                        no_gridlines=False,
                        foreground_grid=True,
                    )
                    dpg.add_plot_axis(
                        dpg.mvYAxis,
                        label="mean",
                        tag="plot_cone_ay",
                        no_gridlines=False,
                        foreground_grid=True,
                    )
                    dpg.add_bar_series(
                        [0, 1, 2],
                        [0.0, 0.0, 0.0],
                        weight=0.45,
                        label="cones",
                        parent="plot_cone_ay",
                        tag="series_cone_bars",
                    )
                dpg.add_spacer(height=6)
                dpg.add_text("Opponent means over time (last 80 ticks)")
                dpg.add_checkbox(
                    label="SymLog Y (signed opponent signals)",
                    tag="plot_log_oppo_y",
                    default_value=False,
                    callback=_scale_cb,
                )
                with dpg.plot(height=170, width=-1, tag="plot_oppo_ts", anti_aliased=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(
                        dpg.mvXAxis,
                        label="tick",
                        tag="plot_oppo_ax",
                        no_gridlines=False,
                        foreground_grid=True,
                    )
                    dpg.add_plot_axis(
                        dpg.mvYAxis,
                        label="signal",
                        tag="plot_oppo_ay",
                        no_gridlines=False,
                        foreground_grid=True,
                    )
                    dpg.add_line_series([], [], label="L-M", parent="plot_oppo_ay", tag="series_oppo_lm")
                    dpg.add_line_series([], [], label="S - (L+M)", parent="plot_oppo_ay", tag="series_oppo_by")

            with dpg.tab(label="Raster"):
                dpg.add_text("Spike raster (when Poisson spikes are enabled in the left column)")
                with dpg.group(tag="spike_raster_plot_group", show=state.config.spike_output.enabled):
                    with dpg.plot(height=240, width=-1, tag="plot_rgc_raster", anti_aliased=True):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(
                            dpg.mvXAxis,
                            label="time (s)",
                            tag="plot_raster_ax",
                            no_gridlines=False,
                            foreground_grid=True,
                        )
                        dpg.add_plot_axis(
                            dpg.mvYAxis,
                            label="neuron row",
                            tag="plot_raster_ay",
                            no_gridlines=False,
                            foreground_grid=True,
                        )
                        dpg.add_line_series(
                            [],
                            [],
                            label="spikes",
                            parent="plot_raster_ay",
                            tag="series_rgc_raster",
                        )

            with dpg.tab(label="Export & record"):
                dpg.add_text("Export")
                dpg.add_button(
                    label="Save screenshot (PNG)",
                    width=-1,
                    tag="btn_export_png",
                    callback=lambda: dpg.show_item("file_dialog_png"),
                )
                dpg.add_button(
                    label="Save layer stats (CSV)",
                    width=-1,
                    tag="btn_export_csv",
                    callback=lambda: dpg.show_item("file_dialog_csv"),
                )
                dpg.add_button(
                    label="Save layer grids (.npy)",
                    width=-1,
                    tag="btn_export_npy",
                    callback=lambda: dpg.show_item("file_dialog_npy"),
                )
                dpg.add_separator()
                dpg.add_spacer(height=4)
                dpg.add_text("Session recording")
                dpg.add_text(
                    "Capture frames while the sim runs (folder: session_meta.json + session.npz).",
                    wrap=wrap,
                )
                dpg.add_checkbox(label="Record", tag="rec_enabled", default_value=False)
                dpg.add_input_int(
                    label="Spatial downsample stride",
                    default_value=4,
                    min_value=1,
                    max_value=64,
                    width=SIDE_SPIN_WIDTH,
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
                    width=SIDE_SLIDER_WIDTH,
                    callback=lambda s, a: app._apply_playback_frame(),
                )
                dpg.add_text("", tag="rec_status_text", wrap=wrap)
    return root
