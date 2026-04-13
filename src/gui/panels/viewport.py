"""Center panel: simulation texture viewport."""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from src.gui.app_context import AppContext
from src.gui.layout_constants import (
    VIEWPORT_AREA_TAG,
    VIEWPORT_AREA_THEME_TAG,
    VIEWPORT_IMAGE_TAG,
    VIEWPORT_PANEL_BG_RGB_U8,
    VIEWPORT_TEX_TAG,
)


def build(parent_tag: str, ctx: AppContext) -> str:
    """Create viewport theme + texture child under ``parent_tag``. Returns movable root tag."""
    root = "panel_viewport_root"
    r, g, b = VIEWPORT_PANEL_BG_RGB_U8
    with dpg.group(parent=parent_tag, tag=root):
        with dpg.theme(tag=VIEWPORT_AREA_THEME_TAG):
            with dpg.theme_component(dpg.mvChildWindow):
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (r, g, b, 255))
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (r, g, b, 255))
        with dpg.child_window(
            border=True,
            width=-1,
            height=-1,
            tag=VIEWPORT_AREA_TAG,
            horizontal_scrollbar=True,
            no_scroll_with_mouse=True,
        ):
            with dpg.group(horizontal=False):
                dpg.add_image(VIEWPORT_TEX_TAG, tag=VIEWPORT_IMAGE_TAG, width=400, height=400)
        dpg.bind_item_theme(VIEWPORT_AREA_TAG, VIEWPORT_AREA_THEME_TAG)
    return root
