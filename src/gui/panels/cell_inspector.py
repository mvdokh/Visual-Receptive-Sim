"""
Cell inspector panel: shows full input/output connectivity tree for the picked cell.
Supports RGC, cone, bipolar, horizontal, amacrine. Cone bar chart and convergence summary.
"""

from __future__ import annotations

from typing import Any, Optional

import dearpygui.dearpygui as dpg

from src.simulation.connectivity import (
    AmacrineConnectivityResult,
    BipolarConnectivityResult,
    ConeConnectivityResult,
    HorizontalConnectivityResult,
    RGCConnectivityResult,
)


INSPECTOR_PANEL_TAG = "cell_inspector_panel"
INSPECTOR_HEADER = "inspector_header"
INSPECTOR_INPUTS = "inspector_inputs"
INSPECTOR_CONVERGENCE = "inspector_convergence"
INSPECTOR_OUTPUT = "inspector_output"
INSPECTOR_CONE_CHART = "inspector_cone_chart"


def build_inspector_panel(parent: Optional[int] = None) -> int:
    """Build the inspector panel UI. Returns the group/window tag.
    When parent is None, items are added to the current container (e.g. the active tab)."""
    if parent is not None:
        with dpg.group(tag=INSPECTOR_PANEL_TAG, parent=parent) as g:
            _build_inspector_content()
    else:
        with dpg.group(tag=INSPECTOR_PANEL_TAG) as g:
            _build_inspector_content()
    return g


def _build_inspector_content() -> None:
    """Add inspector widgets to the current group (avoids passing parent=None to DPG)."""
    dpg.add_text("No cell selected. Click a cell in the viewport.", tag=INSPECTOR_HEADER)
    dpg.add_separator()
    with dpg.child_window(height=320, border=True):
        dpg.add_text("", tag="inspector_position")
        dpg.add_text("", tag="inspector_ecc")
        dpg.add_text("", tag="inspector_dendritic")
        dpg.add_separator()
        dpg.add_text("INPUT TREE", color=(200, 220, 255, 255))
        dpg.add_text("", tag=INSPECTOR_INPUTS)
        dpg.add_text("Cone inputs (L / M / S)", tag="inspector_cone_label")
        with dpg.plot(tag=INSPECTOR_CONE_CHART, height=80, width=-1):
            dpg.add_plot_axis(dpg.mvXAxis, tag="inspector_cone_x")
            dpg.add_plot_axis(dpg.mvYAxis, tag="inspector_cone_y")
            dpg.add_bar_series([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], tag="inspector_cone_series", parent="inspector_cone_y", weight=0.6)
        dpg.add_text("", tag=INSPECTOR_CONVERGENCE)
        dpg.add_separator()
        dpg.add_text("OUTPUT", color=(200, 220, 255, 255))
        dpg.add_text("", tag=INSPECTOR_OUTPUT)


def update_inspector_from_rgc(result: RGCConnectivityResult) -> None:
    """Fill inspector panel from RGC connectivity result."""
    if not dpg.does_item_exist(INSPECTOR_HEADER):
        return
    dpg.set_value(
        INSPECTOR_HEADER,
        f"RGC #{result.cell_id}  |  Parasol ON  |  ecc: {result.eccentricity_deg:.1f}°",
    )
    dpg.set_value(
        "inspector_position",
        f"Position: ({result.position_px[0]:.0f}, {result.position_px[1]:.0f}) px  |  "
        f"({result.position_um[0]:.0f}, {result.position_um[1]:.0f}) μm",
    )
    dpg.set_value("inspector_ecc", f"Eccentricity: {result.eccentricity_deg:.2f}°")
    dpg.set_value(
        "inspector_dendritic",
        f"Dendritic field: {result.dendritic_diameter_um:.0f} μm diameter",
    )
    cb = result.cone_breakdown
    dpg.set_value(
        INSPECTOR_INPUTS,
        f"Bipolar cells:  {result.bipolar_on_midget} ON-midget, {result.bipolar_on_diffuse} ON-diffuse\n"
        f"── Cones:  L: {cb.n_L}  M: {cb.n_M}  S: {cb.n_S}  Total: {cb.total}\n"
        f"   (ratio:  {cb.ratio_L*100:.0f}%  {cb.ratio_M*100:.0f}%  {cb.ratio_S*100:.0f}%)\n"
        f"── Rods:  ~{result.rod_count} within rod bipolar range\n"
        f"Amacrine (inhib):  AII ×{result.amacrine_aii}, wide-field ×{result.amacrine_wide}",
    )
    dpg.set_value(
        "inspector_cone_series",
        [[0.0, 1.0, 2.0], [float(cb.n_L), float(cb.n_M), float(cb.n_S)]],
    )
    if dpg.does_item_exist("inspector_cone_y"):
        dpg.set_axis_limits("inspector_cone_y", 0, max(1, cb.total) * 1.1)
    e_lo, e_hi = result.biological_expectation_range
    flag = "✓" if e_lo <= result.total_photoreceptor_inputs <= e_hi else "⚠"
    dpg.set_value(
        INSPECTOR_CONVERGENCE,
        f"Photoreceptors → this RGC:  ~{result.total_photoreceptor_inputs} total\n"
        f"(biological expectation at this ecc:  {e_lo}–{e_hi})  {flag}",
    )
    dpg.set_value(
        INSPECTOR_OUTPUT,
        f"Pathway:  {result.pathway}\nCurrent firing rate:  {result.firing_rate:.1f} sp/s",
    )


def update_inspector_from_cone(result: ConeConnectivityResult) -> None:
    """Fill inspector for a picked cone."""
    if not dpg.does_item_exist(INSPECTOR_HEADER):
        return
    dpg.set_value(
        INSPECTOR_HEADER,
        f"Cone #{result.cell_id}  |  Type: {result.cone_type}  |  ecc: {result.eccentricity_deg:.2f}°",
    )
    dpg.set_value(
        "inspector_position",
        f"Position: ({result.position_px[0]:.0f}, {result.position_px[1]:.0f}) px",
    )
    dpg.set_value("inspector_ecc", f"Eccentricity: {result.eccentricity_deg:.2f}°")
    dpg.set_value("inspector_dendritic", "Photoreceptor (no dendritic field)")
    rgc_list = ", ".join(f"#{i}" for i in result.rgc_ids)
    if result.rgc_count_upstream > len(result.rgc_ids):
        rgc_summary = f"{rgc_list}  …  ({result.rgc_count_upstream} total)"
    elif result.rgc_ids:
        rgc_summary = f"{rgc_list}  ({result.rgc_count_upstream} total)"
    else:
        rgc_summary = f"{result.rgc_count_upstream} RGC(s)"
    dpg.set_value(
        INSPECTOR_INPUTS,
        f"Bipolar cells receiving from this cone:  {result.bipolar_count}\n"
        f"Horizontal cells pooling from this cone:  {result.horizontal_count}\n"
        f"Feeds into RGCs:  {rgc_summary}",
    )
    dpg.set_value(
        "inspector_cone_series",
        [[0.0, 1.0, 2.0], [1.0 if result.cone_type == "L" else 0.0, 1.0 if result.cone_type == "M" else 0.0, 1.0 if result.cone_type == "S" else 0.0]],
    )
    dpg.set_value(INSPECTOR_CONVERGENCE, "")
    dpg.set_value(INSPECTOR_OUTPUT, "Output: drives bipolar and horizontal cells.")


def update_inspector_from_bipolar(result: BipolarConnectivityResult) -> None:
    """Fill inspector for a picked bipolar cell."""
    if not dpg.does_item_exist(INSPECTOR_HEADER):
        return
    dpg.set_value(
        INSPECTOR_HEADER,
        f"Bipolar #{result.cell_id}  |  {result.cell_type}  |  activation: {result.activation:.3f}",
    )
    dpg.set_value(
        "inspector_position",
        f"Position: ({result.position_px[0]:.0f}, {result.position_px[1]:.0f}) px",
    )
    dpg.set_value("inspector_ecc", "")
    dpg.set_value("inspector_dendritic", "")
    dpg.set_value(
        INSPECTOR_INPUTS,
        f"Cone inputs:  {result.cone_count}\n"
        f"Amacrine cells providing inhibition:  {result.amacrine_count}\n"
        f"RGC(s) this bipolar feeds:  {result.rgc_count}",
    )
    dpg.set_value("inspector_cone_series", [[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]])
    dpg.set_value(INSPECTOR_CONVERGENCE, "")
    dpg.set_value(INSPECTOR_OUTPUT, f"Current activation: {result.activation:.4f}")


def update_inspector_from_horizontal(result: HorizontalConnectivityResult) -> None:
    """Fill inspector for a picked horizontal cell."""
    if not dpg.does_item_exist(INSPECTOR_HEADER):
        return
    dpg.set_value(
        INSPECTOR_HEADER,
        f"Horizontal #{result.cell_id}  |  dendritic extent: {result.dendritic_extent_um:.0f} μm",
    )
    dpg.set_value(
        "inspector_position",
        f"Position: ({result.position_px[0]:.0f}, {result.position_px[1]:.0f}) px",
    )
    dpg.set_value("inspector_ecc", "")
    dpg.set_value("inspector_dendritic", f"Spatial extent: {result.dendritic_extent_um:.0f} μm")
    ct = result.cone_types
    dpg.set_value(
        INSPECTOR_INPUTS,
        f"Cones within pooling radius:  {result.cone_count}  (L: {ct.n_L}, M: {ct.n_M}, S: {ct.n_S})",
    )
    dpg.set_value(
        "inspector_cone_series",
        [[0.0, 1.0, 2.0], [float(ct.n_L), float(ct.n_M), float(ct.n_S)]],
    )
    dpg.set_value(INSPECTOR_CONVERGENCE, "")
    dpg.set_value(INSPECTOR_OUTPUT, "Surround contribution to nearby RGCs (indirect).")


def update_inspector_from_amacrine(result: AmacrineConnectivityResult) -> None:
    """Fill inspector for a picked amacrine cell."""
    if not dpg.does_item_exist(INSPECTOR_HEADER):
        return
    dpg.set_value(
        INSPECTOR_HEADER,
        f"Amacrine #{result.cell_id}  |  {result.cell_type}  |  reach: {result.reach_um:.0f} μm",
    )
    dpg.set_value(
        "inspector_position",
        f"Position: ({result.position_px[0]:.0f}, {result.position_px[1]:.0f}) px",
    )
    dpg.set_value("inspector_ecc", "")
    dpg.set_value("inspector_dendritic", f"Inhibitory reach: {result.reach_um:.0f} μm")
    dpg.set_value(
        INSPECTOR_INPUTS,
        f"Cells within inhibitory radius:  RGCs receiving inhibition: {result.rgc_count_inhibited}",
    )
    dpg.set_value("inspector_cone_series", [[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]])
    dpg.set_value(INSPECTOR_CONVERGENCE, "")
    dpg.set_value(INSPECTOR_OUTPUT, "Output: inhibitory synapses onto RGCs and bipolars.")


def update_inspector(result: Any, layer_name: str) -> None:
    """Dispatch to the right updater by layer and result type."""
    if result is None:
        if dpg.does_item_exist(INSPECTOR_HEADER):
            dpg.set_value(INSPECTOR_HEADER, "No cell selected. Click a cell in the viewport.")
        return
    if isinstance(result, RGCConnectivityResult):
        update_inspector_from_rgc(result)
    elif isinstance(result, ConeConnectivityResult):
        update_inspector_from_cone(result)
    elif isinstance(result, BipolarConnectivityResult):
        update_inspector_from_bipolar(result)
    elif isinstance(result, HorizontalConnectivityResult):
        update_inspector_from_horizontal(result)
    elif isinstance(result, AmacrineConnectivityResult):
        update_inspector_from_amacrine(result)
    else:
        if dpg.does_item_exist(INSPECTOR_HEADER):
            dpg.set_value(INSPECTOR_HEADER, f"Unknown cell type: {type(result).__name__}")


def clear_inspector() -> None:
    """Reset inspector to empty state."""
    if dpg.does_item_exist(INSPECTOR_HEADER):
        dpg.set_value(INSPECTOR_HEADER, "No cell selected. Click a cell in the viewport.")
    for tag in ["inspector_position", "inspector_ecc", "inspector_dendritic",
                INSPECTOR_INPUTS, INSPECTOR_CONVERGENCE, INSPECTOR_OUTPUT]:
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, "")
    if dpg.does_item_exist("inspector_cone_series"):
        dpg.set_value("inspector_cone_series", [[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]])
