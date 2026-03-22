Conceptual overview
===================

This page is the **model without the window**: what is computed at each stage, in order. The :doc:`equations` page names the same operations in math; :doc:`/user_guide/interface` maps them to controls and views.

Signal flow
-----------

Light to firing rate is a feedforward chain with horizontal feedback at the cones:

1. **Stimulus** — A spectral radiance grid :math:`S(x, y, \lambda)` over a retinal patch (e.g. 1° × 1°), sampled in wavelength (e.g. 380–700 nm).

2. **Cones** — L, M, S photoreceptor responses via spectral sensitivity (e.g. Stockman & Sharpe 2000 fundamentals), giving :math:`C_L, C_M, C_S` at each location.

3. **Horizontal cells** — Spatially pooled cone signals (L+M and S pathways) with Gaussian kernels; output :math:`H` provides **surround feedback** to cones.

4. **Cone effective output** — Cone signal minus horizontal feedback (center–surround), yielding :math:`C^{\mathrm{eff}}_L, C^{\mathrm{eff}}_M, C^{\mathrm{eff}}_S`.

5. **Bipolar cells** — ON/OFF split (rectification of :math:`C^{\mathrm{eff}}`) and optional diffuse pooling; midget (small receptive field) and parasol (wider) pathways.

6. **Amacrine cells** — AII and wide-field amacrines pool bipolar (and cone) activity; output provides **lateral inhibition** subtracted from bipolar drive.

7. **RGC generators** — Dendritic integration modeled as Gaussian pooling of (bipolar − amacrine) with pathway-specific :math:`\sigma`.

8. **Firing rate** — Linear–nonlinear (LN) model: sigmoid nonlinearity maps generator signal to spike rate (sp/s).

9. **Color opponent** — L−M and S−(L+M) signals for display/analysis.

10. **Temporal dynamics** — First-order RC smoothing per layer (exponential filter with layer-specific :math:`\tau`).

All spatial operations are implemented as 2D convolutions (Gaussian kernels) on a regular grid; the pipeline is vectorized over the grid (no per-cell loops).

Scope
-----

- **Spatial**: Single patch of retina (e.g. 256×256 grid in 1°); periodic or reflected boundary conditions.
- **Spectral**: Monochromatic or multi-wavelength stimuli via cone fundamentals; no explicit opponent wiring before RGCs beyond L−M at the output.
- **Temporal**: Discrete time steps; RC smoothing approximates low-pass filtering at each layer.

For symbols and parameters, see :doc:`equations`; for biological context, see :doc:`biology`.
