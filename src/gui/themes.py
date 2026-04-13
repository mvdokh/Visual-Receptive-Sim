"""Dear PyGui global theme from JSON tokens (simulation colormaps are separate)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dearpygui.dearpygui as dpg


@dataclass
class ThemeTokens:
    window_bg: tuple[int, ...]
    child_bg: tuple[int, ...]
    text: tuple[int, ...]
    text_disabled: tuple[int, ...]
    button: tuple[int, ...]
    frame_bg: tuple[int, ...]
    border: tuple[int, ...]
    separator: tuple[int, ...]
    plot_bg: tuple[int, ...]
    plot_axis: tuple[int, ...]
    plot_line: tuple[int, ...]


def _tup4(d: dict[str, Any], key: str, default: tuple[int, int, int, int]) -> tuple[int, ...]:
    v = d.get(key, list(default))
    if isinstance(v, (list, tuple)) and len(v) >= 4:
        return tuple(int(x) for x in v[:4])
    return default


def load_theme(path: str | Path) -> ThemeTokens:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    d = raw.get("tokens", raw)
    return ThemeTokens(
        window_bg=_tup4(d, "window_bg", (23, 23, 23, 255)),
        child_bg=_tup4(d, "child_bg", (30, 30, 30, 255)),
        text=_tup4(d, "text", (240, 240, 240, 255)),
        text_disabled=_tup4(d, "text_disabled", (128, 128, 128, 255)),
        button=_tup4(d, "button", (51, 51, 55, 255)),
        frame_bg=_tup4(d, "frame_bg", (40, 40, 40, 255)),
        border=_tup4(d, "border", (70, 70, 70, 255)),
        separator=_tup4(d, "separator", (80, 80, 80, 255)),
        plot_bg=_tup4(d, "plot_bg", (22, 22, 22, 255)),
        plot_axis=_tup4(d, "plot_axis", (160, 160, 160, 255)),
        plot_line=_tup4(d, "plot_line", (120, 180, 255, 255)),
    )


def build_dpg_theme(tokens: ThemeTokens) -> int:
    """Create a Dear PyGui theme tag (global chrome + plot defaults)."""
    theme_id = dpg.generate_uuid()
    with dpg.theme(tag=theme_id):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, tokens.window_bg)
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, tokens.child_bg)
            dpg.add_theme_color(dpg.mvThemeCol_Text, tokens.text)
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, tokens.text_disabled)
            dpg.add_theme_color(dpg.mvThemeCol_Button, tokens.button)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, tokens.frame_bg)
            dpg.add_theme_color(dpg.mvThemeCol_Border, tokens.border)
            dpg.add_theme_color(dpg.mvThemeCol_Separator, tokens.separator)
        with dpg.theme_component(dpg.mvPlot):
            dpg.add_theme_color(
                dpg.mvPlotCol_PlotBg, tokens.plot_bg, category=dpg.mvThemeCat_Plots
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_FrameBg, tokens.plot_bg, category=dpg.mvThemeCat_Plots
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_AxisText, tokens.plot_axis, category=dpg.mvThemeCat_Plots
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_Line, tokens.plot_line, category=dpg.mvThemeCat_Plots
            )
    return int(theme_id)


def theme_json_path(theme_id: str) -> Path:
    return Path(__file__).resolve().parent / "themes" / f"{theme_id}.json"
