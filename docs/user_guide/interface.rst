User interface
==============

This page describes the **graphical interface**: panels, views, and how they connect to the model. For math and biology, use the concepts pages (linked below).

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

Layout
------

The window has three main areas:

- **Left panel** — View mode, 2D layer selector (when applicable), stimulus controls, and when **3D Stack** is selected: 3D camera, connectivity toggles, **Inspection layer** (coarse pick for viewport clicks), then **Circuit tuning** (**synaptic weights** and **cell parameters**) in the same scroll area.
- **Center** — The main viewport: 2D heatmap, 2D all-layers mosaic, or the 3D Signal Flow Column (Vispy).
- **Right panel (tabs)** — **Stats**, **Plots**, **Export** (below), and **Inspector** (only when **3D Stack** is active; connectivity for the picked cell).

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
- **Heatmap colormap** selects how scalar grids are painted: Firing, Biphasic, Spectral, Diverging. These are the same modes as ``grid_to_rgba`` in ``src/rendering/heatmap.py`` (see figure below).
- **Stimulus** uses a spectral-derived RGB from the current stimulus spectrum; other layers use the scalar colormaps as chosen.

.. figure:: ../_static/examples/heatmap_colormaps.png
   :align: center
   :width: 70%

   One synthetic 2D grid shown with the four colormap modes (``scripts/generate_doc_example_plots.py`` calls ``grid_to_rgba`` like the viewer).

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

Tree nodes group **pathway** and **pool** knobs (narrow vs wide RGC fields, bipolar pooling, horizontal feedback, narrow vs wide lateral inhibition), not fixed histological names. They map to :math:`\sigma` scales, LN parameters (:math:`r_{\max}`, slope, half-point), and :math:`\tau` values in :doc:`/concepts/equations` (section 7 and the LN figure there).

Stats tab
---------

The **Stats** tab reads the same arrays the pipeline writes each frame (see ``_update_stats`` in ``src/gui/app.py``).

- **Mean FR per RGC type** — Spatial mean of the **midget ON (L)** and **parasol ON** firing-rate grids (``fr_midget_on_L``, ``fr_parasol_on``).
- **L−M and S−(L+M)** — One summary line each: spatial mean of ``lm_opponent`` and ``by_opponent`` (same definitions as :doc:`/concepts/equations`, color-opponent section).
- **Per-layer** — For Stimulus, Cones L/M/S, Horizontal, Bipolar, Amacrine, and RGC: mean, standard deviation, min, and max over the grid. Stimulus uses the spectrally summed mask (sum over wavelength); **RGC** here is the midget ON (L) firing grid.
- **RGC dynamics → RGC mean FR (last 100 ticks)** — A line trace of the spatial mean of ``fr_midget_on_L`` each tick; the window keeps the last 100 samples so you can see drift or oscillation as the stimulus or state changes.
- **RGC FR histogram** — Distribution of **pixel** values in the current ``fr_midget_on_L`` grid (default 16 bins over the data range; the UI updates this plot on a throttled cadence). Figures below use **example data** from a headless ``pipeline.tick`` run (same stimulus and logic as the app).

.. figure:: ../_static/examples/gui_rgc_mean_fr_by_type.png
   :align: center
   :width: 58%

   Example: mean firing rate across the grid for each type (same quantities as the two lines at the top of the Stats tab).

.. figure:: ../_static/examples/gui_rgc_mean_fr_sparkline.png
   :align: center
   :width: 65%

   Example: last 100 ticks of spatial mean **midget ON (L)** firing rate.

.. figure:: ../_static/examples/gui_rgc_fr_histogram.png
   :align: center
   :width: 65%

   Example: histogram of **midget ON (L)** firing rates over space at one time (same layer as the 2D heatmap when that layer is selected).

Plots tab
---------

- **Cone mean drive (L / M / S)** — Three bars: spatial mean of ``cone_L``, ``cone_M``, and ``cone_S`` (raw cone outputs after the spectral inner product, before the horizontal surround step).
- **Opponent means over time (last 80 ticks)** — Two traces: spatial mean of ``lm_opponent`` and ``by_opponent`` each tick, keeping the last 80 samples. **L−M** is built from midget ON L and M firing grids; **S−(L+M)** from effective cone outputs (see :doc:`/concepts/equations`).

.. figure:: ../_static/examples/gui_cone_mean_drive.png
   :align: center
   :width: 52%

   Example: cone mean drive bars (same three means as the Plots tab).

.. figure:: ../_static/examples/gui_opponent_means_timeseries.png
   :align: center
   :width: 65%

   Example: opponent means over the last 80 recorded ticks.

Export tab
----------

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - Control
     - Output
   * - **Save screenshot (PNG)**
     - Captures the **current viewport** RGBA buffer (what you see in the center panel) as a PNG via ``export_screenshot_png`` in ``src/gui/panels/data_export.py``.
   * - **Save layer stats (CSV)**
     - One CSV row per named layer: **mean**, **min**, **max**, **std** over the grid. Layers: ``cone_L``, ``cone_M``, ``cone_S``, ``h_activation``, ``bp_midget_on_L``, ``bp_diffuse_on``, ``amacrine_aii``, ``fr_midget_on_L``, ``fr_parasol_on``, ``lm_opponent``, ``by_opponent``.
   * - **Save layer grids (.npy)**
     - Writes one ``.npy`` file per array into the **directory** you choose. Files match ``SimState`` field names (full 2D grids or stimulus spectrum ``(H, W, L)``). See the list in ``export_layer_grids_npy`` in ``data_export.py``.

Arrays written by **Save layer grids** (when present on ``SimState``):

.. list-table::
   :widths: 22 78
   :header-rows: 1

   * - File
     - Role
   * - ``stimulus_spectrum.npy``
     - Spectral radiance cube :math:`(H, W, L_\lambda)`.
   * - ``cone_L.npy``, ``cone_M.npy``, ``cone_S.npy``
     - Raw cone responses.
   * - ``cone_L_eff.npy``, ``cone_M_eff.npy``, ``cone_S_eff.npy``
     - Cone outputs after horizontal feedback.
   * - ``h_activation.npy``
     - Horizontal cell pool.
   * - ``bp_*.npy``, ``amacrine_*.npy``
     - Bipolar and amacrine layers (ON/OFF and pathways as named in code).
   * - ``fr_*.npy``
     - RGC firing rates (midget / parasol, ON / OFF).
   * - ``lm_opponent.npy``, ``by_opponent.npy``
     - Opponent maps (same as in :doc:`/concepts/equations`).

Figures on this page are generated by ``scripts/generate_doc_example_plots.py`` (``plot_gui_panels_from_pipeline`` uses ``pipeline.tick`` with a drifting full-field grating so the time traces are non-trivial).

For GitHub Pages deployment of this documentation, see :doc:`deploy_github_pages`.
