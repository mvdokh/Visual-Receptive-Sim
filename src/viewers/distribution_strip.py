"""
Distribution strip: dockable panel with per-cell-type activity histograms.

One row per cell type with color swatch, count badge, mini histogram,
visibility toggle, reorder handle. Updates every N simulation frames.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from src.viewers.layer_manager import LAYER_ORDER

# Cell type display name -> (inactive_color_rgb, state_key)
DISTRIBUTION_LAYERS = [
    ("L cone", (0.784, 0.294, 0.192), "cone_L"),
    ("M cone", (0.227, 0.49, 0.267), "cone_M"),
    ("S cone", (0.169, 0.31, 0.78), "cone_S"),
    ("Horizontal", (0.545, 0.412, 0.078), "h_activation"),
    ("ON Bipolar", (0.102, 0.478, 0.29), "bp_diffuse_on"),
    ("AII Amacrine", (0.29, 0.29, 0.541), "amacrine_aii"),
    ("Wide Amacrine", (0.416, 0.165, 0.416), "amacrine_wide"),
    ("Midget RGC", (0.722, 0.525, 0.043), "fr_midget_on_L"),
    ("Parasol RGC", (0.545, 0.102, 0.102), "fr_parasol_on"),
]

HISTOGRAM_BINS = 50
ACTIVE_THRESHOLD = 0.01


def _get_layer_grid(state, key: str) -> Optional[np.ndarray]:
    """Extract (H,W) grid from SimState."""
    return getattr(state, key, None)


def compute_histogram(grid: np.ndarray, bins: int = HISTOGRAM_BINS) -> Tuple[np.ndarray, np.ndarray]:
    """Compute histogram of firing rates. Returns (bin_centers, counts)."""
    flat = grid.flatten().astype(np.float32)
    flat = flat[np.isfinite(flat)]
    if len(flat) == 0:
        return np.zeros(bins), np.zeros(bins)
    counts, edges = np.histogram(flat, bins=bins, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers.astype(np.float32), counts.astype(np.float32)


def count_active(grid: np.ndarray, threshold: float = ACTIVE_THRESHOLD) -> int:
    """Count cells with rate > threshold."""
    if grid is None:
        return 0
    return int(np.sum(grid.flatten() > threshold))


def get_total_cells(grid: np.ndarray) -> int:
    """Total cells in layer."""
    if grid is None:
        return 0
    return int(grid.size)


try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
        QDockWidget, QFrame, QComboBox, QSizePolicy,
    )
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPainter, QColor
    HAS_QT = True
except ImportError:
    try:
        from PyQt6.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
            QDockWidget, QFrame, QComboBox, QSizePolicy,
        )
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QPainter, QColor
        HAS_QT = True
    except ImportError:
        HAS_QT = False

if HAS_QT:
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False
else:
    HAS_MATPLOTLIB = False


class HistogramMini(QWidget):
    """Small embedded histogram (50 bins)."""

    def __init__(self, parent=None, width: int = 80, height: int = 24):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self._centers: np.ndarray = np.zeros(HISTOGRAM_BINS)
        self._counts: np.ndarray = np.zeros(HISTOGRAM_BINS)
        self._color = (0.5, 0.5, 0.5)

    def set_data(self, centers: np.ndarray, counts: np.ndarray, color: Tuple[float, float, float]) -> None:
        self._centers = centers
        self._counts = counts
        self._color = color
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(30, 30, 35))
        if self._counts.max() <= 0:
            p.end()
            return
        w, h = self.width(), self.height()
        scale = h / max(1e-6, float(self._counts.max()))
        c = QColor(int(self._color[0] * 255), int(self._color[1] * 255), int(self._color[2] * 255))
        p.setPen(c)
        p.setBrush(c)
        n = len(self._counts)
        bar_w = max(1, w / n - 1)
        for i in range(n):
            x = i * (w / n)
            bar_h = float(self._counts[i]) * scale
            if bar_h > 0.5:
                p.drawRect(int(x), int(h - bar_h), max(1, int(bar_w)), int(bar_h))
        p.end()


class DistributionRow(QWidget):
    """One row: color swatch, label, count badge, histogram, visibility, layer dropdown."""

    visibility_changed = None  # Signal(bool) - set by parent

    def __init__(
        self,
        label: str,
        color: Tuple[float, float, float],
        state_key: str,
        parent=None,
    ):
        super().__init__(parent)
        self.state_key = state_key
        self.label_text = label
        self.color = color
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 0)

        # Color swatch
        swatch = QFrame()
        swatch.setFixedSize(14, 14)
        swatch.setStyleSheet(
            f"background: rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)});"
            "border: 1px solid #444;"
        )
        layout.addWidget(swatch)

        # Label
        layout.addWidget(QLabel(label), 0, Qt.AlignmentFlag.AlignLeft)

        # Count badge
        self.count_label = QLabel("0/0")
        self.count_label.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(self.count_label)

        # Mini histogram
        self.hist = HistogramMini(self, width=80, height=20)
        layout.addWidget(self.hist)

        # Visibility toggle
        self.vis_check = QCheckBox()
        self.vis_check.setChecked(True)
        self.vis_check.setToolTip("Show/hide this cell type")
        layout.addWidget(self.vis_check)

        # Layer assignment dropdown (placeholder)
        self.layer_combo = QComboBox()
        self.layer_combo.addItems(["Layer 5", "Layer 4", "Layer 3", "Layer 2", "Layer 1"])
        self.layer_combo.setMaximumWidth(80)
        layout.addWidget(self.layer_combo)


class DistributionStripDock(QDockWidget if HAS_QT else object):
    """
    QDockWidget containing the distribution strip.
    Can be docked below the 3D canvas or floated.
    """

    def __init__(self, title: str = "Distribution", parent=None):
        if not HAS_QT:
            raise RuntimeError("PySide6 or PyQt6 required for DistributionStripDock")
        super().__init__(title, parent)
        self._widget = QWidget()
        self._layout = QVBoxLayout(self._widget)
        self._rows: Dict[str, DistributionRow] = {}
        self._frame_counter = 0
        self._refresh_every = 5

        for label, color, state_key in DISTRIBUTION_LAYERS:
            row = DistributionRow(label, color, state_key)
            self._rows[state_key] = row
            self._layout.addWidget(row)

        self.setWidget(self._widget)
        self.setMinimumHeight(120)

    def update_from_state(self, state) -> None:
        """Update histograms and counts. Call every frame; internally throttles to refresh_every."""
        self._frame_counter += 1
        if self._frame_counter % self._refresh_every != 0:
            return
        for state_key, row in self._rows.items():
            grid = _get_layer_grid(state, state_key)
            if grid is None:
                continue
            active = count_active(grid)
            total = get_total_cells(grid)
            row.count_label.setText(f"{active}/{total}")
            centers, counts = compute_histogram(grid)
            row.hist.set_data(centers, counts, row.color)

    def set_refresh_rate(self, every_n_frames: int) -> None:
        self._refresh_every = max(1, every_n_frames)
