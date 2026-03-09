# =============================================================
# Biological constants for the RGC Circuit Simulator
# All values sourced from peer-reviewed human/mammalian retina data.
# Both the 2D heatmap viewer and the 3D Signal Flow viewer import
# from this file so biology is never inconsistent between views.
#
# References for scale parameters:
# - Drasdo & Fowler (1974) Ophthalmic & Physiological Optics: 290 μm/degree
# - Curcio & Allen (1990) J Comp Neurol 300(1):5–25: RGC density by eccentricity
# - Dacey (1993) J Neurosci 13(12):5334–5355: Midget/parasol dendritic scaling
# - Watanabe & Rodieck (1989) J Comp Neurol 289(3):434–454: Parasol field diameter
# =============================================================

# --- Physical scale (retina) ---
# Human foveal cone spacing: ~2.5 μm center-to-center
# Human peripheral cone spacing: ~10–30 μm at 10° eccentricity
# Rod spacing: ~2–3 μm (tightly packed outside fovea)
# Midget RGC dendritic field (fovea): ~5–10 μm diameter
# Parasol RGC dendritic field (5° ecc.): ~100–200 μm diameter
# Parasol RGC dendritic field (20° ecc.): ~300–500 μm diameter
# 1° of visual angle ≈ 290 μm on the human retina (Drasdo & Fowler 1974)
MICRONS_PER_DEGREE = 290.0  # μm/° on human retina
FOVEAL_CONE_SPACING = 2.5  # μm
PERIPHERAL_CONE_SPACING = 15.0  # μm at ~10° eccentricity
ROD_SPACING = 2.5  # μm
MIDGET_DEND_DIAMETER = 7.0  # μm (fovea)
PARASOL_DEND_DIAMETER = 150.0  # μm (mid-periphery)

# --- Large-field grid (recommended: 2048×2048, 4 μm/px ≈ 28° patch) ---
# Current small grid: 256×256 px; at 4 μm/px → 1024 μm ≈ 3.5°
# 2048×2048 px at 4 μm/px → 8192 μm ≈ 28° visual angle
# 4096×4096 px at 4 μm/px → 16384 μm ≈ 56° visual angle
GRID_SIZE_PX = 2048  # pixels per side (large-field default)
MICRONS_PER_PX = 4.0  # μm per pixel
GRID_SIZE_MICRONS = GRID_SIZE_PX * MICRONS_PER_PX  # 8192 μm
GRID_SIZE_DEGREES = GRID_SIZE_MICRONS / MICRONS_PER_DEGREE  # ~28°

# RGC density (Curcio & Allen 1990): parasol ~150 cells/mm² at 10°; midget ~600 at 10°
# Total RGC at foveal center ~35,000 cells/mm²
PARASOL_RGC_DENSITY_PER_MM2 = 150  # at ~10° eccentricity
MIDGET_RGC_DENSITY_PER_MM2 = 600  # at ~10° eccentricity

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
