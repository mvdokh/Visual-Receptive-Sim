"""Session recording save/load round-trip."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.config import default_config
from src.simulation.pipeline import tick
from src.simulation.session_recording import SessionRecordingBuffer, load_session_recording
from src.simulation.state import SimState


def test_session_recording_roundtrip(tmp_path: Path) -> None:
    state = SimState(config=default_config())
    state.ensure_initialized()
    state.stimulus_params.update({"type": "spot", "intensity": 0.7, "radius_deg": 0.1})
    buf = SessionRecordingBuffer(stride=2)
    for _ in range(5):
        tick(state, 0.02)
        buf.append_frame(state)
    out = tmp_path / "sess"
    buf.save(out)

    meta = json.loads((out / "session_meta.json").read_text(encoding="utf-8"))
    assert meta["n_frames"] == 5
    assert meta["stride"] == 2

    loaded = load_session_recording(out)
    assert loaded.n_frames == 5
    state2 = SimState(config=default_config())
    state2.ensure_initialized()
    loaded.apply_frame(2, state2)
    assert abs(float(state2.time) - buf.times[2]) < 1e-6
    assert state2.stimulus_params.get("type") == "spot"
