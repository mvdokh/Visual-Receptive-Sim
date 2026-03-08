## RGC Circuit Simulator — Python Edition

**This is a Python reimplementation of the retinal ganglion cell (RGC) circuit simulator.**  
The original Rust-based prototype now lives in `legacy/` (kept untouched). The active simulator
is the Python package in `rgc_simulator/`, using NumPy/SciPy/Numba for the biology and
ModernGL + Dear PyGui for a real‑time 3D GUI.

---

### What This Is

This is a simulation of the first stage of human vision — **the retina**.

Before your brain ever processes an image, your eye has already done something remarkable:
it has transformed raw light into a highly processed neural signal. The retina is not a
passive camera. It is a piece of your brain, sitting in your eye, performing real computation.

The simulator tracks how **every photon** flows through the first visual circuit:
from photoreceptors, through horizontal, bipolar, and amacrine cells, out through
retinal ganglion cells and into the optic nerve.

---

### The Biological Story

When light enters your eye and hits the back of the retina, it lands on photoreceptors — cone
cells that are selectively sensitive to different wavelengths. You have three types:
**L‑cones** (sensitive to long wavelengths, roughly red), **M‑cones** (medium, roughly green),
and **S‑cones** (short, roughly blue/violet). Their overlapping sensitivities are why you see color at all.

But the signal doesn't travel straight to your brain. It passes through four more layers of neurons
before leaving the eye, and at each layer something is being computed:

- **Horizontal cells** spread laterally across the retina and feed an inhibitory signal back onto
  the cones. This is where the surround of a receptive field comes from — a cell literally
  suppresses its own neighbors.
- **Bipolar cells** collect cone signals and split them into two streams: **ON** cells that increase
  their activity when light hits their region, and **OFF** cells that increase when light is removed.
- **Amacrine cells** perform further lateral inhibition, sharpening signals in time and space.
- **Retinal ganglion cells (RGCs)** collect from all of the above and send the final output
  signal down the optic nerve to the brain.

The result is that by the time any signal leaves your eye, the retina has already performed
**edge detection, contrast enhancement, color opponency, and motion detection** — all before
the brain is involved.

---

### What the Simulator Shows

This tool lets you place light — any color, any shape, anywhere — and watch that causal chain
fire in real time in 3D, using biologically motivated parameters.

You can see:

- **Which cone types** absorb your stimulus and how strongly
- **How horizontal cell feedback** sculpts surround suppression
- **How ON and OFF bipolar streams** split and carry the signal downward
- **How amacrine cells** sharpen the signal laterally
- **Which RGCs fire, at what rate, and why** — with the signal traceable at every layer

The receptive field of a ganglion cell — the region of the visual world it responds to —
is not assumed or manually drawn. It emerges from the circuit. If you weaken the horizontal
cell feedback, the surround shrinks. If you pool more cones into a bipolar cell, the center grows.
The biology produces the math, not the other way around.

---

### Why It Matters

The retina is the most studied neural circuit in the brain, precisely because it is accessible
and its inputs are controllable — you can show it exactly the light you want. Almost everything
we know about how neural circuits compute comes partly from retinal research.

The center‑surround receptive field discovered here in the 1950s and 60s became the conceptual
foundation for convolutional neural networks, edge detection algorithms, and much of modern
computer vision. This simulator sits at the intersection of that biological reality and the
computational models built to describe it.

---

### Core Philosophy

**Every photon has a traceable causal path through six biological layers, visualized in true 3D.**

---

### Technology Stack

- **Simulation**: NumPy + SciPy (+ Numba-ready for JIT/CUDA)
- **Spectral data**: `colour-science` (CIE color matching functions, cone fundamentals)
- **3D rendering**: ModernGL (OpenGL 3.3 core)
- **GUI / panels**: Dear PyGui (immediate‑mode GUI; scrollable panels and floating windows)
- **Image I/O**: Pillow + `imageio`
- **RF fitting**: `scipy.optimize.curve_fit` (planned)
- **Plotting**: Matplotlib embedded in DPG textures (planned)

---

### Directory Structure

At a high level:

```text
rgc_simulator/
│
├── main.py                        ← entry point (in repo root)
├── requirements.txt               ← Python dependencies
│
├── rgc_simulator/
│   ├── __init__.py
│   ├── config.py                  ← biological + numerical constants
│   │
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── state.py               ← SimState dataclass, shared across threads
│   │   ├── pipeline.py            ← master tick(): runs all layers in order
│   │   ├── layers/
│   │   │   ├── cones.py           ← L/M/S spectral response helpers
│   │   │   ├── horizontal.py      ← lateral pooling, cone feedback
│   │   │   ├── bipolar.py         ← ON/OFF midget/diffuse helpers
│   │   │   ├── amacrine.py        ← AII + wide‑field inhibition helpers
│   │   │   └── rgc.py             ← LN model helpers
│   │   └── stimulus/
│   │       └── spectral.py        ← monochromatic / full‑field spectra
│   │
│   ├── rendering/
│   │   ├── __init__.py
│   │   ├── context.py             ← ModernGL context, offscreen framebuffer
│   │   ├── heatmap.py             ← NumPy grid → RGBA texture
│   │   └── scene_3d/
│   │       ├── layer_planes.py    ← floating planes, one per retinal layer
│   │       ├── cell_spheres.py    ← 3D spheres per cell (data structures)
│   │       ├── synaptic_connections.py ← connection primitives
│   │       ├── signal_flow.py     ← animated particle descriptors
│   │       ├── rf_volume.py       ← RF volume data structure
│   │       └── camera.py          ← orbit camera parameters
│   │
│   └── gui/
│       ├── __init__.py
│       ├── app.py                 ← Dear PyGui setup, main loop
│       ├── panels/                ← logical UI panels (stubs wired for growth)
│       └── widgets/               ← reusable widgets (stubs)
│
├── data/
│   ├── cone_fundamentals.csv?     ← (optional) Stockman & Sharpe 2000 data
│   ├── natural_images/            ← example stimuli (user‑supplied)
│   └── exports/                   ← exported RF maps, CSV, screenshots
│
└── legacy/                        ← original Rust prototype (kept untouched)
```

Most of the biological math currently lives in `simulation/pipeline.py`, with small helper
functions in `simulation/layers/*`. Rendering starts in `rendering/context.py`, which draws
layer planes into an offscreen framebuffer that Dear PyGui displays as a texture.

---

### Quick Start

#### 1. Create a virtual environment

```bash
cd Visual-Receptive-Sim
python -m venv .venv
source .venv/bin/activate  # on macOS / Linux
# .venv\Scripts\activate   # on Windows
```

#### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. Run the simulator

```bash
python main.py
```

The first run will download and build the required Python packages. Subsequent runs are fast.

---

### What You See (Current Python Version)

- **Left panel**:  
  Stimulus controls (type, wavelength, intensity, radius) and basic circuit sliders
  (horizontal feedback, amacrine gain, RGC nonlinearity parameters, temporal constants).

- **Center**:  
  A 3D‑like stack of **retinal layer planes**, rendered by ModernGL and embedded into
  Dear PyGui as a single texture. Each plane is a heatmap of activity:
  cones, horizontals, bipolars, amacrines, and RGC firing.

- **Right panel**:  
  Summary statistics — mean firing rate per RGC type and summary of L−M and S−(L+M)
  opponent signals across the field.

As you move the wavelength slider or adjust circuit parameters, the activations across layers
update in real time.

---

### Roadmap / Planned Features

- Full 3D instanced spheres for every simulated cell (cones, horizontals, bipolars,
  amacrines, RGCs), with activity‑dependent glow.
- Explicit synaptic connection visualization (excitatory/inhibitory/gap junctions),
  including “fan” surrounds from horizontal cells.
- Animated signal‑flow particles tracing which pathways carry signal at any moment.
- A receptive‑field volume view for a selected cell, including DoG fits and RF‑derived
  parameters (σ_center, σ_surround, surround ratio).
- Richer GUI panels and widgets (RF inspector, population histograms, spectrum picker).
- RF probe sweep accelerated with Numba (`njit(parallel=True)`).

---

### Notes on `legacy/`

The `legacy/` directory contains the original Rust implementation (e.g. `cargo run`‑based).
It is kept for reference only and is **not modified** by the Python simulator. All new
development happens in the Python package at the repository root.

