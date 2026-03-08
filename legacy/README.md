# RGC Circuit Simulator

A **retinal circuit simulator** where receptive fields **emerge from biology** — no hand-tuned DoG. Center–surround structure comes from horizontal cell feedback, bipolar pooling, and amacrine inhibition.

---

## Quick start (first time)

### 1. Prerequisites

- **Rust** (1.70+). Install from [rustup.rs](https://rustup.rs):
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

### 2. Run the app

From the project folder:

```bash
cargo run
```

The first run may take a minute while dependencies compile. Later runs are much faster.

### 3. What you see

- **Left panel** — View selector, circuit sliders, stimulus type and parameters, and **Statistics** (expand to see firing per cell type).
- **Center** — The current view: stimulus, a single layer (e.g. Cones L, RGC), **All layers** (thumbnails), or **3D stack** (layers as rows of cell dots).
- **Draw stimulus** window — Only when stimulus type is “Draw shape”; click/drag to paint.

### 4. Try this first

1. Leave **View** on **Stimulus**. You should see a colored spot (default: greenish, 550 nm).
2. Change **Stimulus** → **Type** to **Full field λ**. Move the **λ (nm)** slider (380–700). Switch **View** to **Cones L**, **Cones M**, **Cones S** in turn — each cone type responds most at different wavelengths (L ~560 nm, M ~530 nm, S ~420 nm).
3. Set **View** to **3D stack** to see all layers as rows of “cells” (dots).
4. If the UI feels slow, set **Resolution** to **0 (Low)** in the left panel.

---

## Circuit (what’s simulated)

1. **Stimulus** — Light as spectrum (380–700 nm) or multi-color spots.
2. **Cones** — L, M, S with spectral sensitivity (Stockman & Sharpe). Mosaic L:M:S ≈ 10:5:1, randomly placed.
3. **Horizontal cells** — Pool cones laterally; feedback to cones builds the **surround**.
4. **Bipolar cells** — ON (depolarize to light) and OFF (hyperpolarize to light); midget and diffuse types.
5. **Amacrine cells** — Lateral inhibition in the inner plexiform layer.
6. **RGCs** — Sum inputs from bipolars, then sigmoid → firing rate (sp/s). Types: Midget ON/OFF (L and M), Parasol ON/OFF.

---

## Stimulus types

| Type | Use it for |
|------|------------|
| **Spot** | Single color spot (position, radius, wavelength). |
| **White spot** | Broadband spot. |
| **Blinking / Pulsing / Moving spot** | Time-varying stimuli. |
| **Full field λ** | Whole field at one wavelength — best to see L vs M vs S vs wavelength. |
| **Dual spot** | Two spots, two wavelengths (two colors). |
| **Multi spot (4)** | Up to 4 spots, each with its own wavelength. |
| **Draw shape** | Paint your own shape; “Draw stimulus” window appears. |
| **Uniform field** | Constant L, M, S over the field. |

---

## Views

- **Stimulus, Cones L/M/S, Horizontal, Bipolar ON/OFF, Amacrine, RGC, RGC Firing** — One layer at a time (heatmap or firing).
- **All layers** — Grid of small thumbnails for every layer.
- **3D stack** — Each layer as a horizontal band of “cells” (dots) colored by activity.

---

## Tips

- **Wavelength and L/M/S** — Use **Full field λ** and sweep **λ (nm)** to see which cone type (L, M, S) responds. Use **Cones L**, **Cones M**, **Cones S** views.
- **Statistics** — Expand **Statistics** in the left panel to see mean firing per RGC type (Midget ON L, Parasol OFF, etc.) and mean activations for cones, bipolars, horizontals, amacrines.
- **Less lag** — Lower **Resolution** (0 = Low) for a snappier UI.
- **Draw shape** — Choose “Draw shape”, then in the **Draw stimulus** window click or click-and-drag to paint; adjust **Brush** and use **Clear** to reset.

---

## Parameters (left panel)

- **Circuit** — H feedback α, Amacrine γ, RGC r_max, x_half, slope.
- **Temporal** — Cone τ, RGC τ (smoothing for more realistic dynamics).
- **Resolution** — 0 = Low (fast), 1 = Med, 2 = High.

---

## References (in code)

- Cone spectra: Stockman & Sharpe (2000)
- Cone mosaic: Roorda & Williams (1999)
- H-cells: Kamermans & Spekreijse (1999), Dacey et al. (1996)
- Bipolars: Euler & Wässle (1995)
- Amacrine: Masland (2012)
- RGC types: Field & Chichilnisky (2007)
