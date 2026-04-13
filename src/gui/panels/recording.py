"""Session recording and playback controls (widgets now built in ``stats_plots.py``).

Kept for reference; not registered in ``layout.PANEL_BUILDERS``.
"""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from src.gui.app_context import AppContext
from src.gui.layout_constants import (
    RIGHT_PANEL_WIDTH,
    RIGHT_RECORDING_PANEL_HEIGHT,
    SIDE_SLIDER_WIDTH,
    SIDE_SPIN_WIDTH,
)


def build(parent_tag: str, ctx: AppContext) -> str:
    import src.gui.app as app

    rw = int(ctx.shared.get("right_panel_width", RIGHT_PANEL_WIDTH))
    root = "panel_recording_root"
    with dpg.child_window(
        parent=parent_tag,
        tag=root,
        width=rw,
        height=RIGHT_RECORDING_PANEL_HEIGHT,
        border=True,
        autosize_x=False,
    ):
        dpg.add_text("Capture frames while the sim runs (folder: session_meta.json + session.npz).")
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
        dpg.add_text("", tag="rec_status_text", wrap=rw - 28)
    return root
