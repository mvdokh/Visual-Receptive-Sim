# Modular GUI migration notes

## Overview

The Dear PyGui main window is built from **panel modules** under [`src/gui/panels/`](src/gui/panels/), a JSON **layout preset** system in [`src/gui/layout.py`](src/gui/layout.py) + [`src/gui/presets/`](src/gui/presets/), a JSON **color theme** system in [`src/gui/themes.py`](src/gui/themes.py) + [`src/gui/themes/`](src/gui/themes/), and persisted settings in [`src/gui/settings.py`](src/gui/settings.py) (file: `~/.visual_receptive_sim/user_settings.json`).

Simulation logic, widget **tags**, and callback **names** on `SimState` / pipeline code were kept stable; only layout composition moved.

## Panel map

| Panel id | Module | Contents |
|----------|--------|----------|
| `panel_circuit` | [`circuit.py`](src/gui/panels/circuit.py) | **Main viewer** (mode, layer, scale bar, biological scale, heatmap colormap), **stimulus** (bordered `stimulus_section_wrap` + [`build_stimulus_section`](src/gui/panels/stimulus.py): type, sliders, advanced tree, image load — same widget **tags** as before), pathway weights, spike output, detailed cell parameters |
| `panel_heterogeneity` | [`heterogeneity.py`](src/gui/panels/heterogeneity.py) | Spatial heterogeneity (all `sh_*` tags) |
| `panel_viewport` | [`viewport.py`](src/gui/panels/viewport.py) | `panel_viewport_root` group wrapping `viewport_area` + texture image (`rgc_viewport_tex`, etc.) |
| `panel_stats_plots` | [`stats_plots.py`](src/gui/panels/stats_plots.py) | Right column: mean FR / L–M summaries (always), spike raster (`spike_raster_plot_group`, …) when spikes are enabled, plus collapsible trees for extra layer stats, RGC dynamics plots, cone/opponent plots, export buttons, and session recording (`rec_*` tags) |

Legacy split panels [`recording.py`](src/gui/panels/recording.py) and [`raster.py`](src/gui/panels/raster.py) are no longer registered in [`layout.PANEL_BUILDERS`](src/gui/layout.py); their widgets were folded into `panel_stats_plots` so presets should list only `panel_stats_plots` on the right.

**Convention:** **`left_stack`** default is `["panel_circuit", "panel_heterogeneity"]` (`panel_stimulus` is not a layout panel; stimulus UI is embedded in `panel_circuit`). Panels mount inside **`slot_left_stack_scroll`**; stacked sections use **`height=0` + `auto_resize_y=True`** so the column scrolls to heterogeneity. Custom presets that still list `panel_stimulus` will fail to load — remove that id.

## Layout preset schema (`version: 1`)

- **`left_width`**, **`right_width`**: side column widths in px (also stored in `_shared["left_panel_width"]` / `right_panel_width` for the main loop).
- **`slots`**: maps slot name → ordered list of panel ids.
  - **`left_stack`**, **`right_stack`**: vertical stacks (`slot_left_stack`, `slot_right_stack`).
  - **`center`**: usually `["panel_viewport"]` (`slot_center`).
  - **`bottom`**: horizontal band (`slot_bottom`); empty `[]` hides the band. Used by [`plots_bottom.json`](src/gui/presets/plots_bottom.json).

[`LayoutComposer`](src/gui/layout.py) builds each panel **once**; changing preset only runs **`dpg.move_item`** when a panel’s slot changes. If `move_item` fails on your Dear PyGui build, check parent tags and DPG version (see `requirements.txt`: `dearpygui>=1.9`).

## Themes

- Global UI: [`themes.load_theme`](src/gui/themes.py) + [`build_dpg_theme`](src/gui/themes.py) → `dpg.bind_theme`. Does **not** alter simulation heatmap colormaps (`heatmap_colormap_combo`, `_get_heatmap_colormap`, rendering paths).
- Viewport letterbox colors remain the dedicated viewport child theme in [`viewport.py`](src/gui/panels/viewport.py) / [`layout_constants.VIEWPORT_PANEL_BG_RGB_U8`](src/gui/layout_constants.py).

Theme files: [`dark_plus.json`](src/gui/themes/dark_plus.json), [`light.json`](src/gui/themes/light.json), [`high_contrast.json`](src/gui/themes/high_contrast.json), [`paper.json`](src/gui/themes/paper.json) — each has a `tokens` object with RGBA lists.

## Settings file

Path: **`~/.visual_receptive_sim/user_settings.json`**.

Persisted keys: `active_theme_id`, `active_preset`, `last_export_dir`. Unknown keys in the file are ignored on load.

## Adding a new panel

1. Add `src/gui/panels/my_panel.py` with `def build(parent_tag: str, ctx: AppContext) -> str`.
2. Register in [`layout.PANEL_BUILDERS`](src/gui/layout.py) with a stable id (e.g. `panel_mything`).
3. Reference the id in preset JSON under the appropriate slot list.
4. Prefer **deferred** `import src.gui.app as app` inside `build()` if you need helpers from `app.py`, to avoid import cycles.

## Tests

No automated tests import `src.gui.app` today. If you add GUI tests, install Dear PyGui in the test environment. Core tests under `tests/` should remain unaffected.
