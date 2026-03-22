# RGC Circuit Simulator

![RGC circuit simulator UI](images/screenshot1.png)

![RGC circuit simulator 3D view](images/screenshot2.png)

Retinal ganglion cell circuit simulator: stimulus → cones → horizontals → bipolars → amacrines → RGCs. Vectorized NumPy/SciPy pipeline, Cython hotspots, Vispy 3D viewer (embedded in main app), Dear PyGui.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cd hot_numerical
python setup.py build_ext --inplace
cd ..
python main.py
```

The Cython extensions in `hot_numerical/` are required for smooth 60 FPS (Gaussian convolution, stimulus fill, layer updates). Build them once after cloning or after a clean checkout.

## What it does

- **Simulation**: L/M/S cone spectral response, horizontal surround, ON/OFF bipolar, amacrine inhibition, LN RGC nonlinearity. 256×256 grid.
- **Stimuli**: Spot, full-field, annulus, bar, grating, checkerboard. Monochromatic via cone fundamentals (Stockman & Sharpe 2000).
- **Visualization**: 2D heatmap per layer or 3D stack (Vispy). Click a cell in the viewport to inspect connectivity (RGC, bipolar, amacrine, horizontal).
- **Export**: PNG screenshot, CSV stats, NPY layer grids.

## Stack

NumPy, SciPy, Cython, Vispy, Dear PyGui, colour-science, Pillow, scikit-image.

## Layout

```
src/
├── config.py           # Biological constants, layer z-positions
├── simulation/
│   ├── pipeline.py     # Master tick(), vectorized layer updates
│   ├── state.py        # SimState dataclass
│   ├── layers/         # cones, horizontal, bipolar, amacrine, rgc
│   ├── stimulus/       # spectral.py (spot, bar, grating, etc.)
│   └── rf_probe.py     # Probe sweep, DoG fit
├── rendering/
│   ├── context.py      # ModernGL FBO, render_3d()
│   ├── heatmap.py      # Grid → RGBA colormaps
│   └── scene_3d/       # slabs, connectivity, cell_spheres, camera
├── viewers/            # Vispy 3D viewer (viewer_3d, circuit_tracer, layer_manager)
└── gui/
    ├── app.py          # Dear PyGui main loop, panels
    └── panels/         # data_export, cell_inspector

hot_numerical/          # Cython extensions (convolve_2d, layer_update, stimulus_fill, rf_probe_sweep)
docs/                   # Sphinx documentation
tests/                  # pytest suite, bench_performance.py
```

TODO:
- update documentation
- Traceback (most recent call last):
  File "/Users/martindokholyan/Desktop/Visual-Receptive-Sim/src/gui/app.py", line 867, in <lambda>
    callback=lambda s, a, k=key: (_set_conn_weight(state, k, a), _set_connectivity_dirty()),
                                  ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^
  File "/Users/martindokholyan/Desktop/Visual-Receptive-Sim/src/gui/app.py", line 813, in _set_conn_weight
    setattr(state.config.connectivity_weights, key, max(0.0, min(3.0, value)))
    ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: attribute name must be string, not 'NoneType'
- fix oscilloscope plane
- fix scale bar plane in 3d viewer
- fix cell activity contrast in 3d viewer
- fix circuit selector
- Spiking neurons instead of rate coding. This unlocks ISI analysis, burst detection, and adaptation phenomena that the rate model completely erases. Libraries: Brian2 or NEST
- Ribbon synapse dynamics at the cone pedicle
Cones don't just pass signal linearly — their ribbon synapses have a releasable vesicle pool that depletes and recovers. Adding a simple depression model (Tsodyks-Markram) here would give you light adaptation and contrast gain control
- ON/OFF subunit structure within parasol RGCs
Parasol RGCs don't actually pool bipolar input linearly — they have nonlinear subunits (the Enroth-Cugell & Robson Y-cell model). Adding this would reproduce the well-known spatial frequency doubling response that distinguishes magnocellular from parvocellular pathways
- Spike train export to NWB format
The Neurodata Without Borders (NWB) format is the standard for sharing electrophysiology data. If you add spiking, exporting to NWB would let your simulated data be loaded directly into tools like pynapple, MountainSort, or Elephant for spike sorting and analysis


