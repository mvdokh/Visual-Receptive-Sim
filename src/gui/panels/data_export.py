"""Data export panel: PNG screenshot, CSV, NumPy arrays."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.simulation.state import SimState


def export_screenshot_png(rgba: np.ndarray, filepath: Path) -> None:
    """Save an (H, W, 4) uint8 RGBA array as PNG."""
    try:
        from PIL import Image
        img = Image.fromarray(rgba, mode="RGBA")
        img.save(filepath)
    except ImportError:
        import imageio
        imageio.imwrite(str(filepath), rgba)


def export_layer_grids_csv(state: "SimState", filepath: Path) -> None:
    """Export mean firing rates and layer statistics to CSV."""
    rows = ["layer,mean,min,max,std"]
    for name in [
        "cone_L", "cone_M", "cone_S", "h_activation",
        "bp_midget_on_L", "bp_diffuse_on", "amacrine_aii",
        "fr_midget_on_L", "fr_parasol_on", "lm_opponent",
    ]:
        arr = getattr(state, name, None)
        if arr is not None:
            rows.append(f"{name},{np.mean(arr):.6f},{np.min(arr):.6f},{np.max(arr):.6f},{np.std(arr):.6f}")
    filepath.write_text("\n".join(rows))


def export_layer_grids_npy(state: "SimState", dirpath: Path) -> None:
    """Export all layer grids as .npy files in the given directory."""
    dirpath.mkdir(parents=True, exist_ok=True)
    for name in [
        "stimulus_spectrum", "cone_L", "cone_M", "cone_S",
        "cone_L_eff", "cone_M_eff", "cone_S_eff",
        "h_activation", "bp_midget_on_L", "bp_midget_off_L",
        "bp_midget_on_M", "bp_midget_off_M",
        "bp_diffuse_on", "bp_diffuse_off",
        "amacrine_aii", "amacrine_wide",
        "fr_midget_on_L", "fr_midget_off_L", "fr_midget_on_M", "fr_midget_off_M",
        "fr_parasol_on", "fr_parasol_off",
        "lm_opponent", "by_opponent",
    ]:
        arr = getattr(state, name, None)
        if arr is not None:
            np.save(dirpath / f"{name}.npy", arr)
