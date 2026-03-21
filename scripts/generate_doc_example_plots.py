#!/usr/bin/env python3
"""
Generate static figures for Sphinx (docs/_static/examples/).

Run from the repository root:

    python scripts/generate_doc_example_plots.py

Requires: numpy, matplotlib. Loads ``heatmap.py`` via importlib (no OpenGL).
Connectivity curves use a **reduced 1D** chain with the same weight placement as ``pipeline.tick``
(see docstrings in the figure captions).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# Repo root (parent of scripts/)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import importlib.util

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load heatmap.py directly so this script does not import src.rendering (which pulls OpenGL).
_heatmap_path = ROOT / "src" / "rendering" / "heatmap.py"
_spec = importlib.util.spec_from_file_location("heatmap_docplots", _heatmap_path)
_heatmap_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_heatmap_mod)
grid_to_rgba = _heatmap_mod.grid_to_rgba

OUT = ROOT / "docs" / "_static" / "examples"


def _ensure_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def plot_heatmap_colormaps() -> None:
    """Same synthetic blob through grid_to_rgba colormaps used in the 2D viewer."""
    h, w = 128, 128
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2.0, h / 2.0
    sigma = 18.0
    g = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma**2)).astype(np.float32)
    g = g + 0.25 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * (sigma * 1.8) ** 2)).astype(np.float32)

    names = ["firing", "spectral", "diverging", "biphasic"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    for ax, name in zip(axes, names):
        cmap = name if name != "firing" else "firing"
        rgba = grid_to_rgba(g, colormap=cmap, biphasic_center=0.0)  # type: ignore[arg-type]
        ax.imshow(np.clip(rgba[..., :3], 0, 1), origin="lower", aspect="equal")
        ax.set_title(name.replace("_", " ").title(), fontsize=11)
        ax.axis("off")
    fig.suptitle("2D heatmap colormaps (same underlying grid)", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(OUT / "heatmap_colormaps.png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_ln_sigmoid() -> None:
    """RGC LN sigmoid (same form as :doc:`/concepts/equations`, section 7)."""
    x = np.linspace(-3, 3, 400)
    r_max, x_half, slope = 120.0, 0.0, 2.0
    r = r_max / (1.0 + np.exp(-slope * (x - x_half)))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    axes[0].plot(x, r, color="#c45c00", lw=2)
    axes[0].set_title("Baseline (r_max=120, slope=2, x_half=0)")
    axes[0].set_xlabel("generator G (a.u.)")
    axes[0].set_ylabel("firing rate R (sp/s)")
    axes[0].grid(True, alpha=0.3)

    for rmax in (40.0, 120.0):
        axes[1].plot(x, rmax / (1.0 + np.exp(-slope * (x - x_half))), label=f"R_max={rmax:.0f}")
    axes[1].set_title("Varying R_max (ceiling)")
    axes[1].set_xlabel("generator G (a.u.)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    for sl in (1.0, 3.0):
        axes[2].plot(x, r_max / (1.0 + np.exp(-sl * (x - x_half))), label=f"beta={sl}")
    axes[2].set_title("Varying slope beta")
    axes[2].set_xlabel("generator G (a.u.)")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(r"LN model: $R = R_{\max}/(1+e^{-\beta(G-G_{1/2})})$", fontsize=12, y=1.05)
    plt.tight_layout()
    fig.savefig(OUT / "ln_sigmoid_family.png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_cone_fundamentals_and_basis() -> None:
    """
    (1) Normalized L/M/S fundamentals vs wavelength — discrete dot product in code matches
        :math:`C_k = \\sum_\\lambda S(\\lambda)\\bar{k}(\\lambda)` (see equations §1).
    (2) Legacy image RGB Gaussian bases b_R, b_G, b_B (same formulas as ``spectral.py``).
    (3) Cone responses to a narrowband stimulus as peak wavelength scans (max-normalized narrowband).
    """
    from src.config import SpectralConfig

    spec = SpectralConfig()
    lam = np.asarray(spec.wavelengths, dtype=float)

    basis_R = np.exp(-0.5 * ((lam - 610.0) / 15.0) ** 2)
    basis_G = np.exp(-0.5 * ((lam - 540.0) / 15.0) ** 2)
    basis_B = np.exp(-0.5 * ((lam - 450.0) / 15.0) ** 2)
    for b in (basis_R, basis_G, basis_B):
        m = float(np.max(b))
        if m > 0:
            b /= m

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.2), sharex=True)

    axes[0].plot(lam, spec.sens_L, color="#c41e3a", lw=2.0, label=r"$\bar{l}$ (L)")
    axes[0].plot(lam, spec.sens_M, color="#228b22", lw=2.0, label=r"$\bar{m}$ (M)")
    axes[0].plot(lam, spec.sens_S, color="#1e4fa0", lw=2.0, label=r"$\bar{s}$ (S)")
    axes[0].set_ylabel("sensitivity (norm.)")
    axes[0].set_title("Cone fundamentals (Stockman & Sharpe–style grid; see data/cone_fundamentals.csv)")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(lam, basis_R, color="#a02020", lw=2.0, label=r"$b_R$ (610 nm)")
    axes[1].plot(lam, basis_G, color="#208020", lw=2.0, label=r"$b_G$ (540 nm)")
    axes[1].plot(lam, basis_B, color="#2020a0", lw=2.0, label=r"$b_B$ (450 nm)")
    axes[1].set_ylabel("basis (max = 1)")
    axes[1].set_title(r"Image stimulus: $S \approx R\,b_R + G\,b_G + B\,b_B$ (legacy Gaussian basis)")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    peaks = np.arange(380, 701, 2, dtype=float)
    cL, cM, cS = [], [], []
    for lam0 in peaks:
        s = np.exp(-0.5 * ((lam - lam0) / 6.0) ** 2)
        sm = float(np.max(s))
        if sm > 0:
            s = s / sm
        cL.append(float(np.sum(s * spec.sens_L)))
        cM.append(float(np.sum(s * spec.sens_M)))
        cS.append(float(np.sum(s * spec.sens_S)))
    axes[2].plot(peaks, cL, color="#c41e3a", lw=2.0, label=r"$C_L$")
    axes[2].plot(peaks, cM, color="#228b22", lw=2.0, label=r"$C_M$")
    axes[2].plot(peaks, cS, color="#1e4fa0", lw=2.0, label=r"$C_S$")
    axes[2].set_xlabel("peak wavelength of narrowband stimulus (nm)")
    axes[2].set_ylabel(r"$\sum_\lambda S(\lambda)\,\bar{k}(\lambda)$ (arb.)")
    axes[2].set_title("Cone outputs vs monochromatic peak (same inner product as the pipeline)")
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Stimulus and cone responses (equations §1)", fontsize=13, y=0.995)
    plt.tight_layout()
    fig.savefig(OUT / "cone_fundamentals_and_basis.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _reduced_1d_mean_fr(
    *,
    cone_to_horizontal: float,
    cone_to_bipolar: float,
    horizontal_to_cone: float,
    bipolar_to_amacrine: float,
    amacrine_to_bipolar: float,
    bipolar_to_rgc: float,
) -> float:
    """
    1D strip model using the same multiplicative weight roles as ``pipeline.tick``
    (cone → H pool → H feedback → bipolars → amacrines → RGC generator → LN).
    Not identical to the full 2D simulation, but shows how each :math:`w` enters the chain.
    """
    def gaussian_filter1d_np(a: np.ndarray, sigma: float) -> np.ndarray:
        rad = int(max(1, min(200, round(4.0 * sigma))))
        x = np.arange(-rad, rad + 1, dtype=np.float64)
        k = np.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
        k /= np.sum(k)
        p = np.pad(a.astype(np.float64), (rad, rad), mode="reflect")
        return np.convolve(p, k, mode="valid")

    sigma_px = 14.0
    alpha_lm = alpha_s = 0.7
    gm_aii = 0.6
    gm_wide = 0.3

    def G(a: np.ndarray) -> np.ndarray:
        return gaussian_filter1d_np(a, sigma_px)

    n = 512
    x = np.linspace(-1.0, 1.0, n)
    # Smooth positive "spot" cone drive (L/M co-localized; weak S)
    c_l = 0.88 * np.exp(-((x / 0.11) ** 2))
    c_m = 0.80 * np.exp(-((x / 0.11) ** 2))
    c_s = 0.14 * np.exp(-((x / 0.11) ** 2))

    c_lm_in = (c_l + c_m) * cone_to_horizontal
    c_s_in = c_s * cone_to_horizontal
    h_lm = G(c_lm_in)
    h_s = G(c_s_in)
    h_act = alpha_lm * horizontal_to_cone * h_lm + alpha_s * horizontal_to_cone * h_s

    c_le = np.maximum(0.0, c_l - alpha_lm * h_act)
    c_me = np.maximum(0.0, c_m - alpha_lm * h_act)
    c_se = np.maximum(0.0, c_s - alpha_s * h_act)

    bp_on_l = np.maximum(0.0, cone_to_bipolar * c_le)
    bp_on_m = np.maximum(0.0, cone_to_bipolar * c_me)
    bp_diff = np.maximum(0.0, cone_to_bipolar * (c_le + c_me))

    aii = G((bp_on_l + bp_on_m) * bipolar_to_amacrine)
    aw = G((c_le + c_me) * bipolar_to_amacrine)
    a_tot = amacrine_to_bipolar * (gm_aii * aii + gm_wide * aw)

    drive = np.maximum(0.0, bp_on_l - a_tot)
    g_m = G(drive * bipolar_to_rgc)

    r_max, x_half, slope = 120.0, 0.0, 4.0
    fr = r_max / (1.0 + np.exp(-slope * (g_m - x_half)))
    return float(np.mean(fr))


def plot_connectivity_weight_sensitivity() -> None:
    """
    Mean (1D) RGC firing proxy vs each weight in [0, 3] with others at 1 — illustrates
    :math:`w_{C\\to H},\\ldots,w_{B\\to R}` in the equations / :ref:`parameters`.
    """
    keys = [
        ("cone_to_horizontal", r"$w_{C\to H}$ scales cone $\rightarrow$ horizontal pool"),
        ("cone_to_bipolar", r"$w_{C\to B}$ scales cone $\rightarrow$ bipolar drive"),
        ("horizontal_to_cone", r"$w_{H\to C}$ scales horizontal $\rightarrow$ cone feedback"),
        ("bipolar_to_amacrine", r"$w_{B\to A}$ scales bipolar $\rightarrow$ amacrine"),
        ("amacrine_to_bipolar", r"$w_{A\to B}$ scales amacrine $\rightarrow$ bipolar inhibition"),
        ("bipolar_to_rgc", r"$w_{B\to R}$ scales bipolar $\rightarrow$ RGC generator"),
    ]

    ws = np.linspace(0.0, 3.0, 31)
    base = dict(
        cone_to_horizontal=1.0,
        cone_to_bipolar=1.0,
        horizontal_to_cone=1.0,
        bipolar_to_amacrine=1.0,
        amacrine_to_bipolar=1.0,
        bipolar_to_rgc=1.0,
    )

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.8))
    for ax, (k, title) in zip(np.ravel(axes), keys):
        ys = []
        for w in ws:
            kw = dict(base)
            kw[k] = float(w)
            ys.append(_reduced_1d_mean_fr(**kw))
        ax.plot(ws, ys, color="#2c5282", lw=2.0)
        ax.set_xlim(0, 3)
        ax.set_xlabel("weight")
        ax.set_ylabel("mean output (1D proxy, sp/s)")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "How each connectivity weight affects a reduced 1D chain (same factors as the pipeline)\n"
        "Other weights = 1; curves illustrate the role of each symbol in equations / parameters.",
        fontsize=11.5,
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(OUT / "connectivity_weight_sensitivity.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    _ensure_out()
    plot_heatmap_colormaps()
    plot_ln_sigmoid()
    plot_cone_fundamentals_and_basis()
    plot_connectivity_weight_sensitivity()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
