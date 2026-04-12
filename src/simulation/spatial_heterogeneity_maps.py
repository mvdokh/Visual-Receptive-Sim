# =============================================================
# Build and maintain cached spatial maps for heterogeneity modes.
# =============================================================
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.spatial import cKDTree

from src.config import (
    EccentricityGradientType,
    GlobalConfig,
    MosaicLayoutType,
    SpatialHeterogeneityMode,
)
from src.simulation.bio_constants import (
    CURCIO_ECCENTRICITY_DEG_KNOTS,
    CURCIO_RGC_DENSITY_REL_KNOTS,
    EMPIRICAL_ECCENTRICITY_AVAILABLE,
)
from src.simulation.rgc_population import default_type_fractions, normalize_type_fractions
from src.simulation.rgc_type_constants import FUNCTIONAL_GROUPS, RGC_TYPES
from src.simulation.state import SimState

# Functional groups (keys of FUNCTIONAL_GROUPS) → six UI buckets:
# 0 Midget ON, 1 Midget OFF, 2 Parasol ON, 3 Parasol OFF, 4 Bistratified, 5 Other
FUNCTIONAL_GROUP_TO_SPATIAL_BUCKET: Dict[str, int] = {
    "ON_sustained": 0,
    "OFF_sustained": 1,
    "ON_transient": 2,
    "OFF_transient": 3,
    "ON_OS": 4,
    "ON_OFF_small_RF": 4,
    "DS": 5,
    "SbC_other": 5,
}

N_SPATIAL_BUCKETS: int = 6
ECCENTRICITY_RF_SCALE_BINS: int = 24

assert set(FUNCTIONAL_GROUP_TO_SPATIAL_BUCKET.keys()) == set(FUNCTIONAL_GROUPS.keys())


def type_name_to_spatial_bucket(type_name: str) -> int:
    g = str(RGC_TYPES[type_name]["functional_group"])
    return int(FUNCTIONAL_GROUP_TO_SPATIAL_BUCKET[g])


def normalize_six(
    fr: Tuple[float, float, float, float, float, float],
) -> np.ndarray:
    a = np.array(fr, dtype=np.float64)
    a = np.maximum(a, 0.0)
    s = float(a.sum())
    if s <= 1e-12:
        a[:] = 1.0 / N_SPATIAL_BUCKETS
    else:
        a /= s
    return a


def bucket_targets_to_type_fractions(
    bucket_fracs_6: np.ndarray,
) -> Dict[str, float]:
    """Spread six bucket targets across 42 types using registry default proportions within each bucket."""
    base = default_type_fractions()
    bucket_sum = np.zeros(N_SPATIAL_BUCKETS, dtype=np.float64)
    for n, f in base.items():
        b = type_name_to_spatial_bucket(n)
        bucket_sum[b] += float(f)
    out: Dict[str, float] = {}
    for n, f in base.items():
        b = type_name_to_spatial_bucket(n)
        s = bucket_sum[b]
        out[n] = float(bucket_fracs_6[b] * (f / s)) if s > 1e-12 else 0.0
    return normalize_type_fractions(out)


def type_fractions_for_connectivity_coloring(cfg: GlobalConfig) -> Dict[str, float]:
    sh = cfg.spatial_heterogeneity
    if sh.mode != SpatialHeterogeneityMode.TYPE_MAP:
        return default_type_fractions()
    fr6 = normalize_six(sh.type_map.type_fractions)
    return bucket_targets_to_type_fractions(fr6)


def _scatter_noise_map(
    h: int,
    w: int,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if sigma <= 1e-12:
        return np.ones((h, w), dtype=np.float32)
    n = rng.normal(loc=1.0, scale=float(sigma), size=(h, w)).astype(np.float32)
    return np.clip(n, 0.05, 3.0)


def _empirical_density_scale(ecc_deg: np.ndarray) -> np.ndarray:
    ecc = np.asarray(ecc_deg, dtype=np.float64)
    knots = np.array(CURCIO_ECCENTRICITY_DEG_KNOTS, dtype=np.float64)
    vals = np.array(CURCIO_RGC_DENSITY_REL_KNOTS, dtype=np.float64)
    d = np.interp(ecc, knots, vals, left=vals[0], right=vals[-1])
    d0 = float(vals[0])
    inv = np.sqrt(np.maximum(d0 / np.maximum(d, 1e-6), 0.25))
    return inv.astype(np.float64)


def _eccentricity_rf_scale_map(
    h: int,
    w: int,
    fx: float,
    fy: float,
    scale_deg_per_px: float,
    gradient: EccentricityGradientType,
    strength: float,
) -> np.ndarray:
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float64),
        np.arange(w, dtype=np.float64),
        indexing="ij",
    )
    dist_px = np.sqrt((yy - fy) ** 2 + (xx - fx) ** 2)
    ecc_deg = dist_px * float(scale_deg_per_px)
    k = max(0.0, float(strength))
    if gradient == EccentricityGradientType.LINEAR:
        g = 1.0 + k * ecc_deg
    elif gradient == EccentricityGradientType.SQRT:
        g = 1.0 + k * np.sqrt(np.maximum(ecc_deg, 0.0))
    elif (
        gradient == EccentricityGradientType.EMPIRICAL and EMPIRICAL_ECCENTRICITY_AVAILABLE
    ):
        g = _empirical_density_scale(ecc_deg)
    else:
        g = 1.0 + k * ecc_deg
    return np.asarray(g, dtype=np.float32)


def _quantile_bin_map(rf_scale: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (bin_map HxW int32, rep_scale per bin float64)."""
    flat = rf_scale.ravel()
    order = np.argsort(flat)
    n = flat.size
    edges = np.linspace(0, n, n_bins + 1, dtype=int)
    bin_map = np.zeros(n, dtype=np.int32)
    rep = np.ones(n_bins, dtype=np.float64)
    for b in range(n_bins):
        lo, hi = int(edges[b]), int(edges[b + 1])
        if lo >= hi:
            continue
        idx = order[lo:hi]
        bin_map[idx] = b
        rep[b] = float(np.median(flat[idx]))
    return bin_map.reshape(rf_scale.shape), rep


def _place_mosaic_centers(
    n_cells: int,
    w: int,
    h: int,
    mtype: MosaicLayoutType,
    jitter_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return (N, 2) array of (y, x) in pixel coordinates."""
    n = int(max(1, min(2000, n_cells)))
    if mtype == MosaicLayoutType.POISSON:
        y = rng.uniform(0.0, h, size=n)
        x = rng.uniform(0.0, w, size=n)
        return np.column_stack([y, x]).astype(np.float32)

    nx = max(1, int(np.round(np.sqrt(n * w / max(h, 1)))))
    ny = max(1, int(np.ceil(n / nx)))
    xs = np.linspace(0.5, w - 0.5, nx, dtype=np.float64)
    ys = np.linspace(0.5, h - 0.5, ny, dtype=np.float64)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    pts = np.column_stack([yy.ravel(), xx.ravel()])
    if pts.shape[0] > n:
        idx = rng.choice(pts.shape[0], size=n, replace=False)
        pts = pts[idx]
    elif pts.shape[0] < n:
        extra = n - pts.shape[0]
        y2 = rng.uniform(0.0, h, size=extra)
        x2 = rng.uniform(0.0, w, size=extra)
        pts = np.vstack([pts, np.column_stack([y2, x2])])
    pts = pts.astype(np.float32)

    if mtype == MosaicLayoutType.HEX_REGULAR:
        return pts

    # HEX_JITTER: offset alternating rows for hex-ish feel + Gaussian jitter
    j = float(max(0.0, jitter_sigma)) * min(w, h) * 0.02
    pts[:, 0] += rng.normal(0.0, j, size=pts.shape[0]).astype(np.float32)
    pts[:, 1] += rng.normal(0.0, j, size=pts.shape[0]).astype(np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0.0, h - 1e-3)
    pts[:, 1] = np.clip(pts[:, 1], 0.0, w - 1e-3)
    return pts


def rebuild_spatial_heterogeneity(state: SimState, cfg: GlobalConfig) -> None:
    """Fill cached maps on state; clear unused maps. Caller sets heterogeneity_dirty False after."""
    h, w = state.grid_shape()
    sh = cfg.spatial_heterogeneity
    mode = sh.mode

    # Defaults: inactive
    state.scatter_cone_to_bipolar = None
    state.scatter_bipolar_to_rgc = None
    state.scatter_amacrine_to_bipolar = None
    state.type_map = None
    state.eccentricity_rf_scale_map = None
    state.eccentricity_bin_map = None
    state.eccentricity_bin_rep_scale = None
    state.voronoi_cell_id = None
    state.voronoi_centers_xy = None
    state.mosaic_n_cells = 0

    if mode == SpatialHeterogeneityMode.HOMOGENEOUS:
        return

    if mode == SpatialHeterogeneityMode.SCATTER:
        rng = np.random.default_rng(int(sh.scatter.resample_seed))
        sig = float(sh.scatter.sigma)
        if sh.scatter.affect_cone_to_bipolar:
            state.scatter_cone_to_bipolar = _scatter_noise_map(h, w, sig, rng)
        else:
            state.scatter_cone_to_bipolar = np.ones((h, w), dtype=np.float32)
        if sh.scatter.affect_bipolar_to_rgc:
            state.scatter_bipolar_to_rgc = _scatter_noise_map(h, w, sig, rng)
        else:
            state.scatter_bipolar_to_rgc = np.ones((h, w), dtype=np.float32)
        if sh.scatter.affect_amacrine_to_bipolar:
            state.scatter_amacrine_to_bipolar = _scatter_noise_map(h, w, sig, rng)
        else:
            state.scatter_amacrine_to_bipolar = np.ones((h, w), dtype=np.float32)
        return

    if mode == SpatialHeterogeneityMode.TYPE_MAP:
        rng = np.random.default_rng(int(sh.type_map.map_seed))
        p = normalize_six(sh.type_map.type_fractions)
        state.type_map = rng.choice(
            N_SPATIAL_BUCKETS, size=(h, w), p=p.astype(np.float64)
        ).astype(np.int16)
        return

    if mode == SpatialHeterogeneityMode.ECCENTRICITY:
        ec = sh.eccentricity
        cx = float(ec.fovea_px_x)
        cy = float(ec.fovea_px_y)
        # Treat (128,128) as "use grid center" for any resolution
        if abs(cx - 128.0) < 1e-6 and abs(cy - 128.0) < 1e-6:
            cx = (w - 1) * 0.5
            cy = (h - 1) * 0.5
        grad = ec.gradient
        if grad == EccentricityGradientType.EMPIRICAL and not EMPIRICAL_ECCENTRICITY_AVAILABLE:
            grad = EccentricityGradientType.LINEAR
        rf = _eccentricity_rf_scale_map(
            h,
            w,
            cx,
            cy,
            ec.eccentricity_scale_deg_per_px,
            grad,
            ec.rf_growth_strength,
        )
        state.eccentricity_rf_scale_map = rf
        bmap, rep = _quantile_bin_map(np.asarray(rf), ECCENTRICITY_RF_SCALE_BINS)
        state.eccentricity_bin_map = bmap
        state.eccentricity_bin_rep_scale = rep.astype(np.float32)
        return

    if mode == SpatialHeterogeneityMode.MOSAIC:
        rng = np.random.default_rng(int(sh.mosaic.mosaic_seed))
        centers = _place_mosaic_centers(
            sh.mosaic.n_cells,
            w,
            h,
            sh.mosaic.mosaic_type,
            sh.mosaic.jitter_sigma,
            rng,
        )
        state.voronoi_centers_xy = centers
        state.mosaic_n_cells = int(centers.shape[0])
        tree = cKDTree(centers)
        yy, xx = np.meshgrid(
            np.arange(h, dtype=np.float32),
            np.arange(w, dtype=np.float32),
            indexing="ij",
        )
        pts = np.column_stack([yy.ravel(), xx.ravel()])
        _, lab = tree.query(pts)
        state.voronoi_cell_id = lab.astype(np.int32).reshape(h, w)
        return
