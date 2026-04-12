"""
Session recording: capture simulation frames for playback and light-weight analysis.

Format: directory with session_meta.json + session.npz (times, means, optional downsampled RGC map).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.simulation.state import SimState

RECORDING_VERSION = 1

# Scalar summaries per frame (fixed order for npz columns)
MEAN_KEYS = [
    "mean_fr_midget_on_L",
    "mean_cone_L",
    "mean_cone_M",
    "mean_h_activation",
]


def _stim_params_for_json(params: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in params.items():
        if k == "image_mask":
            if v is None:
                out[k] = None
            else:
                arr = np.asarray(v)
                out[k] = {"shape": list(arr.shape), "dtype": str(arr.dtype), "stored": False}
        elif isinstance(v, (bool, int, float, str)) or v is None:
            out[k] = v
        else:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = str(v)
    return out


@dataclass
class SessionRecordingBuffer:
    """In-memory append-only recording."""

    times: List[float] = field(default_factory=list)
    means: List[List[float]] = field(default_factory=list)
    stim_params: List[Dict[str, Any]] = field(default_factory=list)
    fr_L_downsample: List[np.ndarray] = field(default_factory=list)
    stride: int = 4
    grid_shape: Optional[tuple[int, int]] = None

    def clear(self) -> None:
        self.times.clear()
        self.means.clear()
        self.stim_params.clear()
        self.fr_L_downsample.clear()
        self.grid_shape = None

    def append_frame(self, state: SimState) -> None:
        h, w = state.grid_shape()
        if self.grid_shape is None:
            self.grid_shape = (h, w)
        self.times.append(float(state.time))
        fr = state.fr_midget_on_L
        if fr is not None:
            self.means.append(
                [
                    float(np.mean(fr)),
                    float(np.mean(state.cone_L)) if state.cone_L is not None else 0.0,
                    float(np.mean(state.cone_M)) if state.cone_M is not None else 0.0,
                    float(np.mean(state.h_activation)) if state.h_activation is not None else 0.0,
                ]
            )
            s = max(1, int(self.stride))
            fr_ds = np.asarray(fr[::s, ::s], dtype=np.float32)
            self.fr_L_downsample.append(fr_ds.copy())
        else:
            self.means.append([0.0, 0.0, 0.0, 0.0])
            self.fr_L_downsample.append(np.zeros((1, 1), dtype=np.float32))
        self.stim_params.append(_stim_params_for_json(dict(state.stimulus_params)))

    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "version": RECORDING_VERSION,
            "stride": self.stride,
            "grid_shape": list(self.grid_shape or (0, 0)),
            "mean_keys": MEAN_KEYS,
            "n_frames": len(self.times),
            "stim_params": self.stim_params,
        }
        (out_dir / "session_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        t = np.asarray(self.times, dtype=np.float64)
        m = np.asarray(self.means, dtype=np.float32)
        if not self.fr_L_downsample:
            stack = np.zeros((0, 1, 1), dtype=np.float32)
        else:
            stack = np.stack(self.fr_L_downsample, axis=0)
        np.savez_compressed(out_dir / "session.npz", times=t, means=m, fr_midget_L_ds=stack)


def load_session_recording(out_dir: Path) -> LoadedSessionRecording:
    out_dir = Path(out_dir)
    meta_path = out_dir / "session_meta.json"
    npz_path = out_dir / "session.npz"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    data = np.load(npz_path)
    times = np.asarray(data["times"], dtype=np.float64)
    means = np.asarray(data["means"], dtype=np.float32)
    fr_ds = np.asarray(data["fr_midget_L_ds"], dtype=np.float32)
    return LoadedSessionRecording(
        version=int(meta.get("version", 1)),
        stride=int(meta.get("stride", 1)),
        grid_shape=tuple(int(x) for x in meta.get("grid_shape", (0, 0))),
        times=times,
        means=means,
        stim_params=list(meta.get("stim_params", [])),
        fr_midget_L_ds=fr_ds,
    )


@dataclass
class LoadedSessionRecording:
    version: int
    stride: int
    grid_shape: tuple[int, int]
    times: np.ndarray
    means: np.ndarray
    stim_params: List[Dict[str, Any]]
    fr_midget_L_ds: np.ndarray

    @property
    def n_frames(self) -> int:
        return int(self.times.shape[0])

    def apply_frame(self, index: int, state: SimState) -> None:
        """Restore stimulus params and paint downsampled RGC map into full grid (nearest upsample)."""
        if self.n_frames <= 0:
            return
        i = int(np.clip(index, 0, self.n_frames - 1))
        sp = dict(self.stim_params[i])
        # Do not restore image_mask from recording (not stored)
        if "image_mask" in sp and isinstance(sp["image_mask"], dict):
            sp.pop("image_mask", None)
        if "image_mask" not in sp:
            state.stimulus_params.pop("image_mask", None)
        state.stimulus_params.update(sp)
        state.time = float(self.times[i])
        h, w = state.grid_shape()
        state.ensure_initialized()
        ds = self.fr_midget_L_ds[i]
        s = max(1, self.stride)
        if ds.size > 1 and ds.shape[0] > 0 and ds.shape[1] > 0:
            up = np.repeat(np.repeat(ds, s, axis=0), s, axis=1)
            up = up[:h, :w]
            if up.shape != (h, w):
                temp = np.zeros((h, w), dtype=np.float32)
                temp[: up.shape[0], : up.shape[1]] = up
                up = temp
            if state.fr_midget_on_L is not None:
                state.fr_midget_on_L[:] = up
