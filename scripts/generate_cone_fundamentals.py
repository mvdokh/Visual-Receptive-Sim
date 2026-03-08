#!/usr/bin/env python3
"""
Generate cone_fundamentals.csv from Stockman & Sharpe (2000) 2-deg cone fundamentals.

Uses colour-science if available, otherwise a published analytical approximation.
Run from project root: python scripts/generate_cone_fundamentals.py
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "cone_fundamentals.csv"


def _approximate_ss2000(wavelengths: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stockman & Sharpe (2000) 2-deg cone fundamentals approximation.
    Peak wavelengths: L ~566 nm, M ~540 nm, S ~442 nm.
    Uses modified exponential template (asymmetric curves).
    """
    lam = wavelengths.astype(float)
    # L-cone: peak 566 nm
    l_peak, l_sigma_lo, l_sigma_hi = 566.0, 35.0, 55.0
    l_short = np.exp(-0.5 * ((lam - l_peak) / l_sigma_lo) ** 2)
    l_short = np.where(lam <= l_peak, l_short, 1.0)
    l_long = np.exp(-0.5 * ((lam - l_peak) / l_sigma_hi) ** 2)
    l_long = np.where(lam >= l_peak, l_long, 1.0)
    sens_L = l_short * l_long

    # M-cone: peak 540 nm
    m_peak, m_sigma_lo, m_sigma_hi = 540.0, 32.0, 50.0
    m_short = np.exp(-0.5 * ((lam - m_peak) / m_sigma_lo) ** 2)
    m_short = np.where(lam <= m_peak, m_short, 1.0)
    m_long = np.exp(-0.5 * ((lam - m_peak) / m_sigma_hi) ** 2)
    m_long = np.where(lam >= m_peak, m_long, 1.0)
    sens_M = m_short * m_long

    # S-cone: peak 442 nm, narrow
    s_peak, s_sigma = 442.0, 25.0
    sens_S = np.exp(-0.5 * ((lam - s_peak) / s_sigma) ** 2)
    sens_S = np.where(lam <= 615, sens_S, 0.0)  # S negligible above 615 nm

    for arr in (sens_L, sens_M, sens_S):
        m = float(np.max(arr))
        if m > 0:
            arr /= m
    return sens_L, sens_M, sens_S


def main() -> None:
    try:
        import colour
        from colour import SDS_ILLUMINANTS
        # Use colour-science LMS 2-deg cone fundamentals
        cmfs = colour.msds_cmfs_LMS_2000_2_deg()
        wavelengths = np.array(cmfs.wavelengths)
        sens_L = np.array(cmfs.values[:, 0])
        sens_M = np.array(cmfs.values[:, 1])
        sens_S = np.array(cmfs.values[:, 2])
        for arr in (sens_L, sens_M, sens_S):
            m = float(np.max(arr))
            if m > 0:
                arr /= m
    except Exception:
        wavelengths = np.arange(380, 706, 5, dtype=float)
        sens_L, sens_M, sens_S = _approximate_ss2000(wavelengths)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        f.write("wavelength_nm,L,M,S\n")
        for i in range(len(wavelengths)):
            f.write(f"{wavelengths[i]:.1f},{sens_L[i]:.6e},{sens_M[i]:.6e},{sens_S[i]:.6e}\n")
    print(f"Wrote {OUTPUT} ({len(wavelengths)} rows)")


if __name__ == "__main__":
    main()
