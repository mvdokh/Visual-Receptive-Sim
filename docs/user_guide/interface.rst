User interface
==============

This page describes the **graphical interface** only. For the biological and mathematical background, see :doc:`/concepts/overview`, :doc:`/concepts/biology`, and :doc:`/concepts/equations`.

Layout
------

The window has three main areas:

- **Left panel** — View mode (2D Heatmap / 3D Stack), layer selector (2D), stimulus controls, circuit parameters (horizontal, amacrine, RGC, temporal), and for 3D: camera, slice position, connectivity toggles, layer visibility.
- **Center** — The main viewport: either a 2D heatmap of the selected layer or the 3D Signal Flow Column.
- **Right panel (tabs)** — Stats (mean firing rates, L−M / S−(L+M), per-layer stats, RGC sparkline and histogram), Export (PNG, CSV, NPY), Connectivity (weight matrix, Reset, Randomize), Receptive Field (probe sweep, DoG fit).

2D Heatmap mode
---------------

- **Layer** combo selects which layer to display (Stimulus, Cones L/M/S, Horizontal, Bipolar ON, Amacrine, RGC Firing (L)).
- The heatmap uses a firing colormap (black → amber → white) for activation layers; stimulus uses a spectral-derived RGB.

3D Stack (Signal Flow Column) mode
----------------------------------

- **Show signal flow** — Toggles connectivity lines (cone→horizontal, cone→bipolar, bipolar→amacrine, bipolar→RGC). Each type can be toggled separately under “Connectivity types”.
- **Slice position** — Moves the slice used for the per-layer oscilloscope traces (strips to the right of each slab).
- **Camera** — Azimuth, elevation, distance; mouse drag to orbit, scroll to zoom. **View** menu: Top / Front / Isometric presets.
- **Layer visibility** — Checkboxes and opacity sliders for each slab (Stimulus, Cones, Horizontal, Bipolar, Amacrine, RGC).
- **Per-layer trace strips** — To the right of each slab: rolling heatmap (space × time) plus an oscilloscope-style line (horizontal = space, vertical = activity).

Connectivity tab (right panel)
------------------------------

- Editable weights: Cone→Horizontal, Cone→Bipolar, Horizontal→Cone, Bipolar→Amacrine, Amacrine→Bipolar, Bipolar→RGC (range 0–3).
- **Reset to defaults** sets all to 1.0; **Randomize** sets each to a random value in [0.5, 2.0]. Changes apply to both the simulation pipeline and the 3D connectivity lines.

Receptive field probe
---------------------

- Choose RGC type (e.g. midget_on_L), then **Compute RF (24×24 sweep)**. The app runs a probe sweep and fits a difference-of-Gaussians (DoG) model; sigma_center, sigma_surround, and ratio are displayed.

For deploying the documentation (including this user guide) to GitHub Pages, see :doc:`deploy_github_pages`.
