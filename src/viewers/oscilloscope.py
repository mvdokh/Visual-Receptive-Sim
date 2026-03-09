from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class RollingBuffer:
    """Fixed-length rolling buffer for scalar time series."""

    capacity: int = 200

    def __post_init__(self) -> None:
        self._data = np.zeros((self.capacity,), dtype=np.float32)
        self._size = 0
        self._idx = 0

    def append(self, value: float) -> None:
        self._data[self._idx] = float(value)
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def clear(self) -> None:
        self._size = 0
        self._idx = 0

    def values(self) -> np.ndarray:
        """Return values in chronological order as a 1D float32 array."""
        if self._size == 0:
            return np.zeros((0,), dtype=np.float32)
        if self._size < self.capacity:
            return self._data[: self._size].copy()
        # Wrap-around
        idx = self._idx
        return np.concatenate(
            [self._data[idx:], self._data[:idx]],
        ).astype(np.float32, copy=False)


class OscilloscopeRenderer:
    """
    Lightweight NumPy oscilloscope renderer: draws traces into an RGBA image.

    - No matplotlib, no per-pixel Python loops.
    - Uses simple line rasterization in NumPy, optionally with an overlay trace.
    """

    def __init__(self, width_px: int = 120, height_px: int = 40) -> None:
        self.width = int(width_px)
        self.height = int(height_px)

    def _draw_background(self, img: np.ndarray) -> None:
        """Fill background, border, and grid lines."""
        img[...] = np.array([0.039, 0.039, 0.039, 0.85], dtype=np.float32)  # #0A0A0A
        # Grid lines at 0%, 50%, 100% amplitude
        for frac in (0.0, 0.5, 1.0):
            y = int(round((1.0 - frac) * (self.height - 1)))
            if 0 <= y < self.height:
                img[y, :, :3] = 0.102  # #1A1A1A
                img[y, :, 3] = 0.9
        # Border
        img[0, :, :3] = 0.2
        img[-1, :, :3] = 0.2
        img[:, 0, :3] = 0.2
        img[:, -1, :3] = 0.2

    def _trace_coords(
        self,
        values: np.ndarray,
        vmin: float,
        vmax: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map time series values to integer (x, y) coords across the panel width.
        """
        n = values.shape[0]
        if n <= 1:
            xs = np.arange(self.width, dtype=np.int32)
            ys = np.full_like(xs, self.height // 2)
            return xs, ys

        # Use the last self.width samples; interpolate if needed.
        if n > self.width:
            src_x = np.linspace(0.0, 1.0, num=n, dtype=np.float32)
            dst_x = np.linspace(0.0, 1.0, num=self.width, dtype=np.float32)
            vals = np.interp(dst_x, src_x, values.astype(np.float32))
        else:
            vals = np.zeros((self.width,), dtype=np.float32)
            vals[-n:] = values.astype(np.float32)

        eps = max(1e-6, float(vmax - vmin))
        norm = np.clip((vals - float(vmin)) / eps, 0.0, 1.0)
        xs = np.arange(self.width, dtype=np.int32)
        ys = (1.0 - norm) * float(self.height - 1)
        ys_i = np.clip(ys.astype(np.int32), 0, self.height - 1)
        return xs, ys_i

    def _draw_trace(
        self,
        img: np.ndarray,
        values: np.ndarray,
        color: Sequence[float],
        vmin: float,
        vmax: float,
    ) -> None:
        if values.size == 0:
            return
        xs, ys = self._trace_coords(values, vmin, vmax)
        # Draw vertical segments between successive points for continuity.
        x0 = xs[:-1]
        x1 = xs[1:]
        y0 = ys[:-1]
        y1 = ys[1:]
        for xa, xb, ya, yb in zip(x0, x1, y0, y1):
            if xa == xb:
                y = ya
                img[y, xa, :3] = color[:3]
                img[y, xa, 3] = color[3]
                continue
            y_min = min(ya, yb)
            y_max = max(ya, yb)
            ys_seg = np.arange(y_min, y_max + 1, dtype=np.int32)
            xs_seg = np.linspace(float(xa), float(xb), num=ys_seg.size).astype(np.int32)
            xs_seg = np.clip(xs_seg, 0, self.width - 1)
            ys_seg = np.clip(ys_seg, 0, self.height - 1)
            img[ys_seg, xs_seg, :3] = color[:3]
            img[ys_seg, xs_seg, 3] = color[3]

    def render(
        self,
        mean_buffer: RollingBuffer,
        overlay_buffer: Optional[RollingBuffer],
        mean_color: Tuple[float, float, float, float],
        overlay_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ) -> np.ndarray:
        """
        Render oscilloscope image as (H, W, 4) float32 RGBA in [0,1].
        """
        img = np.empty((self.height, self.width, 4), dtype=np.float32)
        self._draw_background(img)

        mean_vals = mean_buffer.values()
        if mean_vals.size == 0:
            return img

        vmin = float(np.min(mean_vals))
        vmax = float(np.max(mean_vals))
        if vmax <= vmin:
            vmax = vmin + 1.0

        # If we have an overlay, dim the mean trace alpha a bit.
        mean_rgba = np.array(mean_color, dtype=np.float32)
        if overlay_buffer is not None and overlay_buffer.values().size > 0:
            mean_rgba[3] *= 0.4

        self._draw_trace(img, mean_vals, mean_rgba, vmin, vmax)

        if overlay_buffer is not None:
            overlay_vals = overlay_buffer.values()
            if overlay_vals.size > 0:
                overlay_rgba = np.array(overlay_color, dtype=np.float32)
                self._draw_trace(img, overlay_vals, overlay_rgba, vmin, vmax)

        return img

