Biological background
=====================

This section gives **conceptual** background on the retinal stages implemented in the simulator. It is independent of the UI.

Retina and layers
-----------------

The vertebrate retina is a layered structure. Light passes through the inner layers to the **photoreceptors** (rods and cones) at the back; signals then flow back toward the **ganglion cells**, whose axons form the optic nerve.

The simulator focuses on the cone-driven pathway in a small patch:

- **Photoreceptors (cones)** — L, M, S cones sample the stimulus spectrum and produce graded (analog) signals. Their spectral sensitivities are given by cone fundamentals (e.g. Stockman & Sharpe 2000).

- **Horizontal cells** — Large receptive fields; they pool cone output and feed back to cones, forming a **center–surround** antagonism (surround suppression).

- **Bipolar cells** — ON and OFF types split the sign of the cone signal (ON: positive; OFF: negative, often via sign-inverting synapses). Midget bipolars have small dendritic fields; diffuse (parasol-path) bipolars pool over a larger area.

- **Amacrine cells** — AII and other amacrines provide **lateral inhibition**: they pool bipolar (and sometimes cone) activity and subtract from the bipolar drive to ganglion cells, shaping temporal and spatial tuning.

- **Retinal ganglion cells (RGCs)** — Integrate bipolar drive minus amacrine inhibition over a dendritic field (Gaussian in the model), then pass through a nonlinearity to produce a **firing rate**. Midget RGCs (small field) and parasol RGCs (larger field) correspond to the midget and parasol pathways in the code.

Center–surround and receptive fields
------------------------------------

- **Center**: Direct cone → bipolar path (after horizontal feedback) gives the “center” of the receptive field.
- **Surround**: Horizontal cells pool over a larger area and subtract from cone output, so the effective cone signal is center minus surround.
- **RGC receptive field**: Shaped by bipolar pooling (and amacrine inhibition); in the model this is a Gaussian kernel. The **receptive field (RF) probe** in the app (e.g. 24×24 sweep, DoG fit) estimates center/surround size and ratio from the simulated RGC response.

Color
-----

- **L, M, S** cones have different spectral sensitivities; the simulator uses L/M/S fundamentals to convert a spectrum to L, M, S responses.
- **L−M** and **S−(L+M)** are computed as color-opponent signals (e.g. at the RGC output and from effective cone signals) for analysis and display; the core pipeline does not implement explicit color-opponent circuitry beyond this.

Parameter names in the code (e.g. :math:`\sigma`, :math:`\alpha`, :math:`\gamma`) correspond to spatial scales and coupling strengths; see :doc:`equations` for their roles in the math.
