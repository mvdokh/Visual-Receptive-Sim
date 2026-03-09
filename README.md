# RGC Circuit Simulator

![RGC circuit simulator UI](images/screenshot1.png)

![RGC circuit simulator 3D view](images/screenshot2.png)

Retinal ganglion cell circuit simulator: stimulus → cones → horizontals → bipolars → amacrines → RGCs. Vectorized NumPy/SciPy pipeline, Cython hotspots, ModernGL 3D rendering, Dear PyGui.

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
- **Visualization**: 2D heatmap per layer or 3D stack. Click a cell in the viewport to inspect connectivity (RGC, bipolar, amacrine, horizontal).
- **Export**: PNG screenshot, CSV stats, NPY layer grids.

## Stack

NumPy, SciPy, Cython, ModernGL, Dear PyGui, colour-science, Pillow, scikit-image.

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
└── gui/
    ├── app.py          # Dear PyGui main loop, panels
    └── panels/         # data_export, cell_inspector

hot_numerical/          # Cython extensions (convolve_2d, layer_update, stimulus_fill, rf_probe_sweep)
docs/                   # Sphinx documentation
tests/                  # pytest suite, bench_performance.py
```
