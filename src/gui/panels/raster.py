"""RGC spike raster (widgets now built in ``stats_plots.py``).

Kept for reference; not registered in ``layout.PANEL_BUILDERS``.
"""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from src.gui.app_context import AppContext
from src.gui.layout_constants import RIGHT_PANEL_WIDTH, RIGHT_RASTER_PANEL_HEIGHT, SIDE_PANEL_TEXT_WRAP


def build(parent_tag: str, ctx: AppContext) -> str:
    state = ctx.sim_state
    rw = int(ctx.shared.get("right_panel_width", RIGHT_PANEL_WIDTH))
    root = "panel_raster_root"
    with dpg.child_window(
        parent=parent_tag,
        tag=root,
        width=rw,
        height=RIGHT_RASTER_PANEL_HEIGHT,
        border=True,
        autosize_x=False,
    ):
        dpg.add_text("RGC spike raster (when Poisson spikes are enabled)")
        dpg.add_text(
            "Same map as 'RGC spikes (L)' heatmap: one row per subsampled cell (row-major, "
            "top of heatmap = high y). Each spike is a vertical tick; simultaneous spikes "
            "are spread across the last sim step in time.",
            wrap=min(SIDE_PANEL_TEXT_WRAP, rw - 28),
        )
        with dpg.group(tag="spike_raster_plot_group", show=state.config.spike_output.enabled):
            with dpg.plot(height=240, width=-1, tag="plot_rgc_raster"):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="time (s)", tag="plot_raster_ax")
                dpg.add_plot_axis(dpg.mvYAxis, label="neuron row", tag="plot_raster_ay")
                dpg.add_line_series(
                    [],
                    [],
                    label="spikes",
                    parent="plot_raster_ay",
                    tag="series_rgc_raster",
                )
    return root
