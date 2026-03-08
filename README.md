# RGC Circuit Simulator

![RGC circuit simulator UI](images/screenshot1.png)

![RGC circuit simulator 3D view](images/screenshot2.png)

Retinal ganglion cell circuit simulator: stimulus → cones → horizontals → bipolars → amacrines → RGCs. Vectorized NumPy/SciPy pipeline, ModernGL 3D rendering, Dear PyGui.

## Quick Start (Python only)

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python main.py
```

## Optional Cython acceleration

The core simulation runs entirely in **NumPy + SciPy**, but there is an optional
`hot_numerical/` package with Cython implementations for numerical hotspots:

- `convolve_2d.pyx` — separable 2D Gaussian pooling
- `layer_update.pyx` — sigmoid LN and temporal RC updates
- `rf_probe_sweep.pyx` — RF probe sweep interpolation

You **do not need** Cython to run the simulator. If you want to enable the
accelerated paths:

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cd hot_numerical
python setup.py build_ext --inplace
cd ..

python main.py
```

Notes:

- The pipeline dynamically imports the Cython extensions when present and falls
  back to pure NumPy/SciPy if they are missing.
- By default, Gaussian pooling still uses SciPy’s highly optimized
  `gaussian_filter`. To **force** the Cython 2D Gaussian implementation, run:

  ```bash
  HOT_NUMERICAL_USE_CYTHON_CONV=1 python main.py
  ```

  (leave this unset unless you specifically want to test the Cython kernel).


## Overview

- **Simulation**: L/M/S cone spectral response, horizontal surround feedback, ON/OFF bipolar split, amacrine lateral inhibition, LN RGC nonlinearity. Grid-based, vectorized.
- **Stimuli**: Spot, full-field, annulus, bar, grating, checkerboard. Monochromatic via cone fundamentals (Stockman & Sharpe 2000).
- **Visualization**: 2D heatmap per layer or 3D stack (planes + cell spheres). Mouse orbit, scroll zoom.
- **RF probe**: 24x24 sweep, DoG fit (sigma_center, sigma_surround, ratio).
- **Export**: PNG screenshot, CSV stats, NPY layer grids.

## Stack

NumPy, SciPy, Numba, ModernGL, Dear PyGui, colour-science, Pillow, scikit-image.

## Layout

```
src/
├── config.py           # Biological constants, layer z-positions
├── simulation/
│   ├── pipeline.py     # Master tick(), vectorized layer updates (+ fast_* wrappers)
│   ├── state.py        # SimState dataclass
│   ├── layers/         # cones, horizontal, bipolar, amacrine, rgc
│   ├── stimulus/       # spectral.py (spot, bar, grating, etc.)
│   └── rf_probe.py     # Probe sweep, DoG fit (+ fast RF path)
├── rendering/
│   ├── context.py      # ModernGL FBO, render_3d()
│   ├── heatmap.py      # Grid → RGBA colormaps
│   └── scene_3d/       # slabs, connectivity, cell_spheres, camera
└── gui/
    ├── app.py          # Dear PyGui main loop, panels
    └── panels/         # data_export

hot_numerical/          # Cython extensions for numerical hotspots (optional)
docs/                   # Sphinx documentation (Read the Docs style)
```

## Tests

Succinct tests for biological assumptions (cone fundamentals, center–surround, ON/OFF, LN sigmoid, temporal RC, stimulus shapes, DoG fit, color opponent) and for config/state/export and slab layout. Run locally:

```bash
pip install -r requirements.txt -r requirements-test.txt
PYTHONPATH=. pytest tests/ -v
```

Tests run automatically on **push** and **pull_request** via GitHub Actions (`.github/workflows/tests.yml`). See **`docs/TESTING.md`** for how to set up and enable testing on GitHub.

## TODO

- Implement multi-color / multi-object stimuli
- Refine 3D viewer
- Add more detailed statistics
- Add connectivity specs
- Refine parameters available when each stimulus is selected
