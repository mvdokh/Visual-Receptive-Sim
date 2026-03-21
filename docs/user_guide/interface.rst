User interface
==============

This page describes the **graphical interface**: panels, views, and how they connect to the model. For math and biology, use the concepts pages (linked below). For runnable plots that mirror the heatmap and LN code, see :doc:`examples`.

Theory cross-references
-----------------------

.. list-table::
   :widths: 38 62
   :header-rows: 1

   * - Topic
     - Where to read more
   * - Signal flow (stimulus → RGC)
     - :doc:`/concepts/overview`
   * - Cell types, center–surround, convergence
     - :doc:`/concepts/biology`
   * - Equations, symbols, parameter list
     - :doc:`/concepts/equations` (:ref:`parameters`)
   * - Plots of colormaps, LN curve, weights (same code paths as the app)
     - :doc:`examples`

Layout
------

The window has three main areas:

- **Left panel** — View mode, 2D layer selector (when applicable), stimulus controls, and when **3D Stack** is selected: 3D camera, connectivity toggles, **Inspection layer** (coarse pick for viewport clicks), then **Circuit tuning** (**synaptic weights** and **cell parameters**) in the same scroll area.
- **Center** — The main viewport: 2D heatmap, 2D all-layers mosaic, or the 3D Signal Flow Column (Vispy).
- **Right panel (tabs)** — **Stats** (summary statistics, RGC sparkline and histogram), **Plots** (cone mean bars, opponent trajectories), **Export** (PNG / CSV / NPY), **Inspector** (only when **3D Stack** is active; connectivity for the picked cell).

View mode
---------

Use **View → Mode** (left panel, **Mode** combo) to switch:

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - Mode
     - Purpose
   * - **2D Heatmap**
     - Single layer, full viewport; use **Layer** to choose which array is shown (:doc:`/concepts/overview`, layers list).
   * - **2D All Layers**
     - Tiled overview of major stages; good for comparing cone vs bipolar vs RGC at a glance.
   * - **3D Stack**
     - Signal-flow column, connectivity lines, per-slab traces; **Inspector** tab becomes available; pick layer applies to viewport clicks.

**Stats**, **Plots**, and **Export** refer to the running simulation state (same pipeline as in :doc:`/concepts/equations`), regardless of view mode.

Image stimuli use the **rgbtolms** RGB→spectrum mapping in the pipeline (no separate UI toggle).

2D Heatmap mode
---------------

- **Layer** combo selects which layer to display (Stimulus, Cones L/M/S, Horizontal, Bipolar ON, Amacrine, RGC Firing (L), etc.).
- **Heatmap colormap** selects how scalar grids are painted: Firing (amber), Biphasic, Spectral, Diverging—see :doc:`examples` for side-by-side plots using ``grid_to_rgba``.
- The heatmap uses a firing colormap (black → amber → white) for activation layers; **Stimulus** uses a spectral-derived RGB from the current stimulus spectrum (cone fundamentals in :doc:`/concepts/biology`).

Image stimuli
-------------

- Selecting **Stimulus type = image** enables **Load image stimulus…**.
- The image is resized to the retinal grid and converted to a per-pixel spectrum before cone integration—see **Pixel-image stimuli** in :doc:`/concepts/equations`.
- **Intensity** scales the spectrum globally (brighter image → stronger cone drive).

3D Stack (Signal Flow Column) mode
----------------------------------

- **Show signal flow** — Toggles connection lines (cone to horizontal, cone to bipolar, bipolar to amacrine, bipolar to RGC). Matches pathway descriptions in :doc:`/concepts/overview`.
- **Slice position** — Moves the slice used for per-layer oscilloscope traces beside each slab.
- **Camera** — Azimuth, elevation, distance; drag to orbit, scroll to zoom. **View** menu: Top / Front / Isometric.
- **Layer visibility** — Per-slab on/off and opacity (Stimulus through RGC).
- **Per-layer trace strips** — Space–time heatmap and line trace for each slab.

**Inspector** tab (right) appears only in this mode: click the viewport to pick a cell; details depend on **Inspection layer (coarse layer)** on the left. Connectivity math follows the same weights as **Circuit tuning** and :ref:`parameters`.

Circuit tuning (left panel, bottom)
-----------------------------------

Synaptic weights
~~~~~~~~~~~~~~~~

Editable weights: Cone to Horizontal, Cone to Bipolar, Horizontal to Cone, Bipolar to Amacrine, Amacrine to Bipolar, Bipolar to RGC (range 0.0–3.0). They scale terms in the pipeline (see weight symbols in :doc:`/concepts/equations`, sections 2–7).

- **Reset weights to 1.0** — Restores defaults.
- **Randomize weights** — Uniform random in [0.5, 2.0] for exploration.

Cell parameters
~~~~~~~~~~~~~~~

Tree nodes group **pathway** and **pool** knobs (narrow vs wide RGC fields, bipolar pooling, horizontal feedback, narrow vs wide lateral inhibition), not fixed histological names. They map to :math:`\sigma` scales, LN parameters (:math:`r_{\max}`, slope, half-point), and :math:`\tau` values in :doc:`/concepts/equations`. The **LN** family plots in :doc:`examples` show how **r_max** and **slope** change the output nonlinearity.

Stats, plots, and export
------------------------

The **Stats** tab lists mean firing, color-opponent summaries, per-layer lines, an RGC sparkline, and a histogram. The **Plots** tab shows mean L/M/S cone drive and recent opponent means. The **Export** tab provides:

- **Save screenshot (PNG)** — Captures the current viewport image.
- **Save layer stats (CSV)** — Tabular summary of layers.
- **Save layer grids (.npy)** — NumPy archive of layer arrays.

For GitHub Pages deployment of this documentation, see :doc:`deploy_github_pages`.
