"""JSON-driven layout shell: slot containers and panel reparenting."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import dearpygui.dearpygui as dpg

from src.gui.app_context import AppContext
from src.gui.layout_constants import (
    ALL_LAYERS_VIEWER_MARGIN_PX,
    LEFT_PANEL_WIDTH,
    LEFT_STACK_SCROLL_TAG,
    MIN_VIEWPORT_WIDTH,
    RIGHT_PANEL_WIDTH,
)
from src.gui.panels import circuit, heterogeneity, stats_plots, viewport

SLOT_ORDER = ("left_stack", "center", "right_stack", "bottom")

PANEL_BUILDERS: dict[str, Callable[[str, AppContext], str]] = {
    "panel_circuit": circuit.build,
    "panel_heterogeneity": heterogeneity.build,
    "panel_viewport": viewport.build,
    "panel_stats_plots": stats_plots.build,
}


def preset_path(name: str) -> Path:
    return Path(__file__).resolve().parent / "presets" / f"{name}.json"


def load_preset_dict(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if int(data.get("version", 1)) != 1:
        raise ValueError(f"Unsupported preset version in {p}")
    return data


def _slot_tag(slot: str) -> str:
    return f"slot_{slot}"


def _mount_parent_for_slot(slot: str) -> str:
    """Where panel roots attach: left column uses a scroll child so stacked panels stay visible."""
    if slot == "left_stack":
        return LEFT_STACK_SCROLL_TAG
    return _slot_tag(slot)


@dataclass
class LayoutComposer:
    """Owns slot item tags and panel root tags; applies presets via ``dpg.move_item``."""

    preset: dict[str, Any]
    ctx: AppContext
    _panel_roots: dict[str, str] = field(default_factory=dict)
    _panel_slot: dict[str, str] = field(default_factory=dict)
    _shell_built: bool = False

    def _apply_widths_to_shared(self) -> None:
        lw = int(self.preset.get("left_width", self.ctx.shared.get("left_panel_width", LEFT_PANEL_WIDTH)))
        rw = int(self.preset.get("right_width", self.ctx.shared.get("right_panel_width", RIGHT_PANEL_WIDTH)))
        self.ctx.shared["left_panel_width"] = lw
        self.ctx.shared["right_panel_width"] = rw
        if dpg.does_item_exist(LEFT_STACK_SCROLL_TAG):
            dpg.configure_item(LEFT_STACK_SCROLL_TAG, width=lw)

    def _panel_to_slot_map(self) -> dict[str, str]:
        out: dict[str, str] = {}
        slots = self.preset.get("slots") or {}
        for slot_name, panel_ids in slots.items():
            for pid in panel_ids:
                if pid in out and out[pid] != slot_name:
                    raise ValueError(f"Panel {pid} assigned to multiple slots")
                out[pid] = slot_name
        return out

    def ensure_shell(self, parent_tag: str) -> None:
        """Create slot container groups once under ``parent_tag``."""
        if self._shell_built:
            return
        slots_cfg = self.preset.get("slots") or {}
        bottom_ids = list(slots_cfg.get("bottom") or [])
        show_bottom = len(bottom_ids) > 0
        lw = int(self.preset.get("left_width", self.ctx.shared.get("left_panel_width", LEFT_PANEL_WIDTH)))
        with dpg.group(parent=parent_tag, tag="layout_root", horizontal=False):
            with dpg.group(tag="layout_middle_row", horizontal=True, height=-1):
                with dpg.group(tag=_slot_tag("left_stack"), horizontal=False):
                    dpg.add_child_window(
                        tag=LEFT_STACK_SCROLL_TAG,
                        width=lw,
                        height=-1,
                        border=False,
                        horizontal_scrollbar=False,
                    )
                with dpg.group(tag=_slot_tag("center"), horizontal=False):
                    pass
                with dpg.group(tag=_slot_tag("right_stack"), horizontal=False):
                    pass
            with dpg.group(tag=_slot_tag("bottom"), horizontal=True, show=show_bottom):
                pass
        self._shell_built = True

    def _first_build_order(self) -> list[str]:
        slots = self.preset.get("slots") or {}
        order: list[str] = []
        seen: set[str] = set()
        for slot in SLOT_ORDER:
            for pid in slots.get(slot, []) or []:
                if pid not in seen:
                    seen.add(pid)
                    order.append(pid)
        return order

    def mount_panels(self) -> None:
        """First-time: build each panel under its slot from the current preset."""
        self._apply_widths_to_shared()
        pmap = self._panel_to_slot_map()
        for pid in self._first_build_order():
            if pid not in PANEL_BUILDERS:
                raise KeyError(f"Unknown panel id: {pid}")
            slot = pmap[pid]
            parent = _mount_parent_for_slot(slot)
            if not dpg.does_item_exist(parent):
                raise RuntimeError(f"Missing slot container {parent}")
            root = PANEL_BUILDERS[pid](parent, self.ctx)
            self._panel_roots[pid] = root
            self._panel_slot[pid] = slot
        self._reorder_panels_within_slots()

    def _reorder_panels_within_slots(self) -> None:
        """Match on-screen sibling order to each slot's panel list (preset JSON order)."""
        slots_cfg = self.preset.get("slots") or {}
        for slot_name, ids in slots_cfg.items():
            if not isinstance(ids, list) or not ids:
                continue
            parent = _mount_parent_for_slot(slot_name)
            if not dpg.does_item_exist(parent):
                continue
            for i, pid in enumerate(ids):
                root = self._panel_roots.get(pid)
                if not root or not dpg.does_item_exist(root):
                    continue
                if i + 1 < len(ids):
                    nxt = self._panel_roots.get(ids[i + 1])
                    if nxt and dpg.does_item_exist(nxt):
                        try:
                            dpg.move_item(root, parent=parent, before=nxt)
                        except Exception:
                            try:
                                dpg.move_item(root, parent=parent)
                            except Exception:
                                pass
                        continue
                try:
                    dpg.move_item(root, parent=parent)
                except Exception:
                    pass

    def apply_preset(self, new_preset: dict[str, Any]) -> None:
        """Update widths and reparent panels whose slot assignment changed."""
        self.preset = new_preset
        self._apply_widths_to_shared()
        pmap = self._panel_to_slot_map()
        slots_cfg = self.preset.get("slots") or {}
        bottom_ids = list(slots_cfg.get("bottom") or [])
        bt = _slot_tag("bottom")
        if dpg.does_item_exist(bt):
            if len(bottom_ids) > 0:
                dpg.configure_item(bt, show=True, height=280)
            else:
                dpg.configure_item(bt, show=False, height=2)

        for pid, slot in pmap.items():
            if pid not in self._panel_roots:
                continue
            root = self._panel_roots[pid]
            if not dpg.does_item_exist(root):
                continue
            parent = _mount_parent_for_slot(slot)
            if self._panel_slot.get(pid) == slot:
                continue
            try:
                if dpg.get_item_parent(root) == parent:
                    self._panel_slot[pid] = slot
                    continue
            except Exception:
                pass
            try:
                dpg.move_item(root, parent=parent)
            except Exception:
                dpg.move_item(root, parent=parent, before=0)
            self._panel_slot[pid] = slot
        self._reorder_panels_within_slots()

    def min_viewport_client_width(self) -> int:
        lw = int(self.ctx.shared.get("left_panel_width", LEFT_PANEL_WIDTH))
        rw = int(self.ctx.shared.get("right_panel_width", RIGHT_PANEL_WIDTH))
        return int(MIN_VIEWPORT_WIDTH + lw + rw + 32)


def composite_pixel_size(grid_h: int, grid_w: int) -> tuple[int, int]:
    """Native 2D All Layers canvas (H, W) in pixels (same formula as app._composite_layout_dims)."""
    label_strip_h = max(12, min(18, int(grid_h * 0.055)))
    row_gap = 2
    comp_w = 3 * int(grid_w)
    comp_h = 4 * int(grid_h) + 4 * label_strip_h + 3 * row_gap
    return comp_h, comp_w


def _all_layers_horizontal_margin_px(
    center_w: float,
    area_h: float,
    tex_w: float,
    tex_h: float,
    comp_w: float,
    comp_h: float,
) -> float:
    """
    Half of (center width − on-screen width of the mosaic) after uniform scale-to-fit,
    matching letterbox + texture scaling in the main viewer (float model).
    """
    tw = comp_w * min(tex_w / max(comp_w, 1e-6), tex_h / max(comp_h, 1e-6))
    scale = min(center_w / max(tex_w, 1e-6), area_h / max(tex_h, 1e-6))
    content_sw = tw * scale
    return (center_w - content_sw) / 2.0


def side_widths_for_all_layers_margin(
    vw: int,
    *,
    tex_w: int,
    tex_h: int,
    comp_h: int,
    comp_w: int,
    base_lw: int,
    base_rw: int,
    min_viewport_w: int,
    area_h: float,
    slack: int = 32,
    margin_px: float = ALL_LAYERS_VIEWER_MARGIN_PX,
) -> tuple[int, int, int]:
    """
    Return (left_w, right_w, center_w) so side columns absorb extra width and the
    center viewer keeps ~margin_px horizontal inset on each side of the mosaic.

    If the window is too narrow to hit the margin target, center_w is maximized
    and sides stay at least base widths.
    """
    min_sum_sides = max(0, base_lw + base_rw)
    max_center = float(max(min_viewport_w, vw - slack - min_sum_sides))
    lo = float(min_viewport_w)
    if max_center <= lo:
        return base_lw, base_rw, int(max_center)

    def margin_at(cw: float) -> float:
        return _all_layers_horizontal_margin_px(
            cw, area_h, float(tex_w), float(tex_h), float(comp_w), float(comp_h)
        )

    m_hi = margin_at(max_center)
    m_lo = margin_at(lo)

    if m_hi < margin_px:
        center_w = int(max_center)
    elif m_lo >= margin_px:
        center_w = int(lo)
    else:
        lo_f, hi_f = lo, max_center
        for _ in range(48):
            mid = (lo_f + hi_f) / 2.0
            if margin_at(mid) >= margin_px:
                hi_f = mid
            else:
                lo_f = mid
        center_w = int(max(lo, min(max_center, hi_f)))

    sum_sides = vw - slack - center_w
    sum_sides = max(min_sum_sides, sum_sides)
    center_w = vw - slack - sum_sides
    extra = sum_sides - min_sum_sides
    if min_sum_sides <= 0:
        lw, rw = base_lw, base_rw
    else:
        el = (extra * base_lw) // min_sum_sides
        lw = base_lw + int(el)
        rw = base_rw + int(extra - el)
    return int(lw), int(rw), int(max(min_viewport_w, center_w))


def apply_center_width_from_viewport(
    viewport_client_width: int,
    viewport_client_height: int,
    shared: dict[str, Any],
) -> None:
    """Resize center ``viewport_area`` to keep at least ``MIN_VIEWPORT_WIDTH``."""
    slack = 32
    lw = int(shared.get("left_panel_width", LEFT_PANEL_WIDTH))
    rw = int(shared.get("right_panel_width", RIGHT_PANEL_WIDTH))
    vw = max(1, int(viewport_client_width))
    center_w = max(MIN_VIEWPORT_WIDTH, vw - lw - rw - slack)
    if dpg.does_item_exist("viewport_area"):
        dpg.configure_item("viewport_area", width=center_w)
