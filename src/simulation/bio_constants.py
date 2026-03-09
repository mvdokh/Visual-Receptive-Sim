# =============================================================
# Biological constants for the RGC Circuit Simulator
# All values sourced from peer-reviewed human/mammalian retina data.
# Both the 2D heatmap viewer and the 3D Signal Flow viewer import
# from this file so biology is never inconsistent between views.
# =============================================================

# --- Photoreceptor counts (Curcio et al., 1990) ---
# Curcio CA et al. J Comp Neurol 292(4):497–523.
# https://doi.org/10.1002/cne.902920402
RODS_TOTAL = 92_000_000
CONES_TOTAL = 4_600_000
ROD_CONE_RATIO = 20.0  # ~20:1 rods to cones

# --- Cone subtype fractions (Curcio et al., 1991) ---
# Curcio CA et al. J Comp Neurol 312(4):610–624.
# https://doi.org/10.1002/cne.903120411
CONE_FRAC_L = 0.64  # L (red-sensitive)
CONE_FRAC_M = 0.32  # M (green-sensitive)
CONE_FRAC_S = 0.02  # S (blue-sensitive)

# --- RGC count and convergence (Masland 2012) ---
# Masland RH. Neuron 76(2):266–280.
# https://doi.org/10.1016/j.neuron.2012.10.002
RGCS_TOTAL = 1_000_000
PHOTORECEPTOR_RGC_RATIO = 100.0  # overall ~100:1
CONE_RGC_RATIO_FOVEA = 1.0  # midget: ~1:1 at fovea
CONE_RGC_RATIO_PERIPHERY = 30.0  # parasol: up to 30:1

# --- Inner nuclear layer proportions (Masland & Raviola, 1998) ---
# Masland RH & Raviola E. Annu Rev Neurosci 23:249–284.
# https://doi.org/10.1146/annurev.neuro.23.1.249
INL_FRAC_BIPOLAR = 0.41
INL_FRAC_AMACRINE = 0.39
INL_FRAC_MULLER = 0.16
INL_FRAC_HORIZONTAL = 0.03

# --- Relative layer densities normalized to RGC = 1 ---
# Use these as display density multipliers in both 2D and 3D viewers.
# Rod:cone ratio ~20:1 — 92M rods, 4.6M cones (Curcio et al. 1990).
# Cone subtypes L:M:S ≈ 64:32:2 (Curcio et al. 1991).
# Overall photoreceptor→RGC convergence ~100:1 (Masland 2012).
# INL: ~41% bipolar, ~39% amacrine, ~3% horizontal (Masland & Raviola 1998).
RELATIVE_DENSITY = {
    "rods": 952,       # 92M / 96.6M total photoreceptors * 1000
    "cones_L": 31,     # 64% of 48 cone units
    "cones_M": 15,     # 32% of 48 cone units
    "cones_S": 2,      # 2% of 48 cone units
    "cones_all": 48,   # total cone units
    "horizontal": 2,   # ~3% of INL; sparse
    "bipolar": 41,      # ~41% of INL; most numerous INL type
    "amacrine": 39,    # ~39% of INL
    "muller": 16,      # ~16% of INL (render only if glia layer shown)
    "rgc": 1,          # convergence reference point
}
