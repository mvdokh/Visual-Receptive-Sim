Model equations
===============

All equations are given in **LaTeX**; the simulator implements these in a discrete, vectorized form on a 2D grid. Parameters are defined in the code (see :ref:`parameters`).

Notation
--------

- :math:`(x,y)` — spatial position on the retinal patch (e.g. in degrees).
- :math:`\lambda` — wavelength; :math:`S(x,y,\lambda)` stimulus spectrum.
- :math:`C_L, C_M, C_S` — L, M, S cone responses; :math:`C^{\mathrm{eff}}` — effective cone output after horizontal feedback.
- :math:`H` — horizontal cell activation; :math:`B` — bipolar; :math:`A` — amacrine; :math:`G` — RGC generator; :math:`R` — firing rate (sp/s).
- :math:`*` — 2D convolution; :math:`G_\sigma` — Gaussian kernel with standard deviation :math:`\sigma` (in degrees or pixels).
- Subscripts ON/OFF and L/M and pathway names (midget, parasol, AII, wide) follow the code.

---

1. Stimulus and cone responses
------------------------------

Stimulus is a spectral grid :math:`S(x,y,\lambda)`. Cone fundamentals :math:`\bar{l}(\lambda), \bar{m}(\lambda), \bar{s}(\lambda)` (normalized) give:

.. math::
   C_L(x,y) = \int S(x,y,\lambda)\,\bar{l}(\lambda)\,d\lambda,\quad
   C_M(x,y) = \int S(x,y,\lambda)\,\bar{m}(\lambda)\,d\lambda,\quad
   C_S(x,y) = \int S(x,y,\lambda)\,\bar{s}(\lambda)\,d\lambda.

In the code this is a discrete dot product over the wavelength axis (einsum).

Pixel-image stimuli
~~~~~~~~~~~~~~~~~~~

For analytic stimuli (spot, bar, grating, etc.), :math:`S(x,y,\lambda)` is factorized into a spatial mask :math:`M(x,y)` and a narrowband spectral profile centered at a
user-chosen wavelength (with an overall intensity gain). For **image** stimuli, the simulator starts from an RGB pixel image :math:`I(x,y) = (R,G,B)` and constructs
an approximate spectrum per pixel before applying the cone fundamentals.

Concretely, three Gaussian "basis" spectra :math:`b_R(\lambda), b_G(\lambda), b_B(\lambda)` are defined, centered at long-, middle-, and short-wave wavelengths
(:math:`\approx 610, 540, 450\,\mathrm{nm}`), normalized so that :math:`\max_\lambda b_\cdot(\lambda) = 1`. Then:

.. math::

   S(x,y,\lambda) \approx R(x,y)\,b_R(\lambda) + G(x,y)\,b_G(\lambda) + B(x,y)\,b_B(\lambda),

with :math:`R,G,B \in [0,1]` the normalized pixel values. A global **Intensity** parameter :math:`\alpha \in [0,1]` multiplies this spectrum,
so the effective stimulus is :math:`\alpha\,S(x,y,\lambda)`. This preserves pixel hue (up to the three-band approximation) so that the cone fundamentals
can bin the image into L/M/S responses in a color-consistent way.

---

2. Horizontal cell pooling
--------------------------

Horizontal cells pool L+M and S separately with Gaussians :math:`G_{\sigma_{\mathrm{LM}}}` and :math:`G_{\sigma_{\mathrm{S}}}`:

.. math::
   H_{\mathrm{LM}} = (C_L + C_M) * G_{\sigma_{\mathrm{LM}}},\qquad
   H_{\mathrm{S}} = C_S * G_{\sigma_{\mathrm{S}}}.

Combined horizontal activation (with weights :math:`\alpha_{\mathrm{LM}}, \alpha_{\mathrm{S}}` and optional connectivity weight :math:`w_{H\to C}`):

.. math::
   H = \alpha_{\mathrm{LM}} w_{H\to C}\, H_{\mathrm{LM}} + \alpha_{\mathrm{S}} w_{H\to C}\, H_{\mathrm{S}}.

---

3. Cone effective output (center–surround)
-----------------------------------------

Horizontal feedback is subtracted from cone output:

.. math::
   C^{\mathrm{eff}}_L = C_L - \alpha_{\mathrm{LM}} H,\quad
   C^{\mathrm{eff}}_M = C_M - \alpha_{\mathrm{LM}} H,\quad
   C^{\mathrm{eff}}_S = C_S - \alpha_{\mathrm{S}} H.

(Optional scaling by :math:`w_{C\to B}` is applied in the bipolar step in the code.)

---

4. Bipolar cells (ON/OFF split)
-------------------------------

Rectification of effective cone output (midget ON/OFF for L and M; optional weight :math:`w_{C\to B}`):

.. math::
   B_{\mathrm{ON},L} = \max(0,\; w_{C\to B}\, C^{\mathrm{eff}}_L),\quad
   B_{\mathrm{OFF},L} = \max(0,\; -w_{C\to B}\, C^{\mathrm{eff}}_L),

and similarly for M. Diffuse ON/OFF use a pooled signal:

.. math::
   P = (C^{\mathrm{eff}}_L + C^{\mathrm{eff}}_M) * G_{\sigma_{\mathrm{diffuse}}},\quad
   B_{\mathrm{diffuse,ON}} = \max(0, P),\quad
   B_{\mathrm{diffuse,OFF}} = \max(0, -P).

---

5. Amacrine lateral inhibition
------------------------------

AII and wide amacrines pool bipolar (and optionally cone) activity with weights :math:`w_{B\to A}`:

.. math::
   A_{\mathrm{AII}} = \bigl( B_{\mathrm{ON},L} + B_{\mathrm{ON},M} \bigr) * G_{\sigma_{\mathrm{AII}}},\qquad
   A_{\mathrm{wide}} = \bigl( C^{\mathrm{eff}}_L + C^{\mathrm{eff}}_M \bigr) * G_{\sigma_{\mathrm{wide}}}.

Total amacrine inhibition (with :math:`\gamma_{\mathrm{AII}}, \gamma_{\mathrm{wide}}` and optional :math:`w_{A\to B}`):

.. math::
   A_{\mathrm{total}} = w_{A\to B}\,\bigl( \gamma_{\mathrm{AII}} A_{\mathrm{AII}} + \gamma_{\mathrm{wide}} A_{\mathrm{wide}} \bigr).

(Code uses :math:`A_{\mathrm{total}}` subtracted from bipolar before the dendritic Gaussian.)

---

6. RGC generators (dendritic integration)
----------------------------------------

Bipolar drive minus amacrine, scaled by :math:`w_{B\to R}`, is convolved with a Gaussian (dendritic field) to get the generator signal:

.. math::
   G = \bigl( B - A_{\mathrm{total}} \bigr) * G_{\sigma_{\mathrm{dend}}}.

Here :math:`B` is the appropriate bipolar pathway (midget ON/OFF L/M or diffuse ON/OFF); :math:`\sigma_{\mathrm{dend}}` is midget or parasol (e.g. :math:`\sigma_{\mathrm{midget}}`, :math:`\sigma_{\mathrm{parasol}}`).

---

7. Firing rate (LN model)
-------------------------

Sigmoid nonlinearity maps generator :math:`G` to firing rate :math:`R` (sp/s):

.. math::
   R = \frac{R_{\max}}{1 + e^{-\beta\,(G - G_{1/2})}}.

Parameters: :math:`R_{\max}` (max rate), :math:`G_{1/2}` (half-max input), :math:`\beta` (slope). In the code these are ``r_max``, ``x_half``, ``slope``.

---

8. Color opponent signals
-------------------------

For analysis/display:

.. math::
   \mathrm{L{-}M} = R_{\mathrm{ON},L} - R_{\mathrm{ON},M},\qquad
   \mathrm{S{-}(L{+}M)} = C^{\mathrm{eff}}_S - \tfrac{1}{2}(C^{\mathrm{eff}}_L + C^{\mathrm{eff}}_M).

---

9. Temporal smoothing (RC filter)
---------------------------------

Each layer is low-pass filtered with a first-order exponential (time constant :math:`\tau`). Discrete update (Euler-like, with step :math:`\Delta t`):

.. math::
   \alpha = \frac{\Delta t}{\tau},\quad
   \alpha \in [0,1],\qquad
   X_{t+1} = X_t + \alpha\,(X^{\mathrm{raw}}_{t+1} - X_t).

So the stored state :math:`X` is the smoothed value; :math:`X^{\mathrm{raw}}` is the new value from the feedforward pipeline. Different :math:`\tau` are used for cone, horizontal, bipolar, amacrine, and RGC layers.

---

.. _parameters:

Parameters (reference)
---------------------

- **Retina**: :math:`\Delta x` (grid spacing in deg), grid size (e.g. 256×256), field size (e.g. 1°).
- **Horizontal**: :math:`\sigma_{\mathrm{LM}}, \sigma_{\mathrm{S}}` (deg), :math:`\alpha_{\mathrm{LM}}, \alpha_{\mathrm{S}}`.
- **Bipolar**: :math:`\sigma_{\mathrm{diffuse}}` (deg).
- **Amacrine**: :math:`\sigma_{\mathrm{AII}}, \sigma_{\mathrm{wide}}` (deg), :math:`\gamma_{\mathrm{AII}}, \gamma_{\mathrm{wide}}`.
- **Dendritic**: :math:`\sigma_{\mathrm{midget}}, \sigma_{\mathrm{parasol}}` (deg).
- **RGC LN**: :math:`R_{\max}`, :math:`G_{1/2}`, :math:`\beta` (slope).
- **Temporal**: :math:`\tau` per layer (cone, horizontal, bipolar, amacrine, RGC).

Connectivity weights :math:`w_{C\to H}, w_{C\to B}, w_{H\to C}, w_{B\to A}, w_{A\to B}, w_{B\to R}` scale the corresponding pathways (default 1); they are used both in the pipeline and in the 3D connectivity visualization.
