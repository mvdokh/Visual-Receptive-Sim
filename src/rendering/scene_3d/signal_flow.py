from __future__ import annotations

"""
Animated signal-flow particles travelling along synaptic connections.
"""

from dataclasses import dataclass

import numpy as np

from .synaptic_connections import Connection


@dataclass
class SignalParticle:
    connection: Connection
    t: float  # 0 → 1 position along connection
    speed: float
    color: np.ndarray  # (4,) RGBA

