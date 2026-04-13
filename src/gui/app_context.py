"""Application context passed into panel `build()` functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.simulation import SimState


@dataclass
class AppContext:
    sim_state: SimState
    textures: dict[str, str]
    shared: dict[str, Any]
