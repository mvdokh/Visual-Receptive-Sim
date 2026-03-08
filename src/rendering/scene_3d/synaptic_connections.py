from __future__ import annotations

"""
Synaptic connection visualization primitives.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np


ConnType = Literal["excitatory", "inhibitory", "gap_junction"]


@dataclass
class Connection:
    source_pos: np.ndarray  # (3,)
    target_pos: np.ndarray  # (3,)
    weight: float
    signal: float
    conn_type: ConnType

