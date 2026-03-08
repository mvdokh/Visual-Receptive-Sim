# Testing

## What is tested

- **Biological assumptions** (see `tests/test_biology.py`): cone fundamentals (Stockman & Sharpe–style normalization and peak wavelengths), cone spectral integration, center–surround subtraction, ON/OFF bipolar rectification, LN sigmoid asymptotes and half-max, temporal RC filter, stimulus spectrum shapes (spot, full-field, image RGB→spectrum), DoG receptive-field fit, and L−M color-opponent sign.
- **Config** (`tests/test_config.py`): `RetinaGeometry.dx_deg`, `layer_z_positions`, `signal_flow_slab_layout`, `ConnectivityWeights`, `GlobalConfig` subconfigs.
- **Heatmap** (`tests/test_heatmap.py`): wavelength→RGB, `spectrum_to_stimulus_rgba`, `_normalize`, `grid_to_rgba` (firing, biphasic, spectral, diverging, unknown raises).
- **Fast conv** (`tests/test_fast_conv.py`): `gaussian_pool_2d` shape, smoothing, constant input.
- **RF probe** (`tests/test_rf_probe.py`): `_dog_2d`, `fit_dog`, `probe_sweep_fast` for all RGC types.
- **Stimulus** (`tests/test_stimulus.py`): all stimulus types (spot, full_field, annulus, bar, grating, moving_spot, dual_spot, image, unknown fallback), params/spectral defaults.
- **State** (`tests/test_state.py`): `grid_shape`, `ensure_initialized` (smoothed, dirty_flags), idempotence.
- **Pipeline** (`tests/test_pipeline.py`): `tick` populates layers, time increment, connectivity weights, smoothed layers updated.
- **Layers** (`tests/test_layers.py`): `compute_cone_responses` shape and sign.
- **Rendering** (`tests/test_rendering.py`): `allocate_trace_buffers`, `_y_center` from slab layout.
- **Config, state, and export** (see `tests/test_ui_ux.py`): default config build, `SimState.ensure_initialized()`, signal-flow slab layout order, all stimulus types returning correct array shape, and export (PNG, CSV, NPY) not crashing and producing valid output.

Tests do **not** start the GUI or open a display; they use NumPy/SciPy and the simulation pipeline only.

## Run tests locally

From the repository root, with the same environment you use for the app:

```bash
pip install -r requirements.txt
pip install -r requirements-test.txt
PYTHONPATH=. pytest tests/ -v
```

Optional: shorter tracebacks with `--tb=short`, or run a single file:

```bash
PYTHONPATH=. pytest tests/test_biology.py -v
```

### Coverage

To measure line coverage of the `src` package:

```bash
PYTHONPATH=. pytest tests/ --cov=src --cov-report=term-missing
```

This prints a summary and lists lines not covered. To generate an HTML report:

```bash
PYTHONPATH=. pytest tests/ --cov=src --cov-report=html
```

Open `htmlcov/index.html` in a browser. Add `pytest-cov` via `pip install -r requirements-test.txt` if needed.

## Set up testing on GitHub

1. **Workflow file**  
   The file `.github/workflows/tests.yml` is already in the repo. It runs the test suite on every **push** and **pull request** to `main` or `master`.

2. **Enable Actions**  
   - On GitHub, open your repo → **Settings** → **Actions** → **General**.  
   - Under “Actions permissions”, choose **Allow all actions and reusable workflows** (or at least allow the default set).  
   - Save.

3. **Trigger a run**  
   - Push a commit to `main`/`master` or open a PR targeting that branch.  
   - Go to the **Actions** tab; you should see a “Tests” workflow run.  
   - Green check = all tests passed; red X = fix the failing tests and push again.

4. **Branch name**  
   If your default branch is not `main` or `master`, edit `.github/workflows/tests.yml` and change the `branches` list under `on.push` and `on.pull_request` to match your branch (e.g. `main`).

No secrets or extra configuration are required; the workflow installs dependencies from `requirements.txt` and `requirements-test.txt` and runs `pytest tests/`.
