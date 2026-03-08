"""
Slice plane: vertical XZ plane at user-controlled X position.
Draws 1D activity profile per layer as a glowing line embedded in 3D.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import moderngl
import numpy as np

from src.config import signal_flow_slab_layout


@dataclass
class SlicePlaneRenderer:
    """Renders 1D activity line graph at slice X position across all layers."""

    ctx: moderngl.Context

    _vao: Optional[moderngl.VertexArray] = field(default=None, init=False)
    _buffer: Optional[moderngl.Buffer] = field(default=None, init=False)

    def _ensure_resources(self) -> None:
        if self._vao is not None:
            return
        prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos;
                in vec3 in_color;
                in float in_bright;
                uniform mat4 u_mvp;
                out vec3 v_color;
                out float v_bright;
                void main() {
                    gl_Position = u_mvp * vec4(in_pos, 1.0);
                    v_color = in_color;
                    v_bright = in_bright;
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 v_color;
                in float v_bright;
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(v_color * v_bright, 0.9);
                }
            """,
        )
        # 6 layers * 256 rows * 2 verts/segment * 7 floats * 4 bytes = ~86KB
        self._buffer = self.ctx.buffer(reserve=256 * 1024)
        self._vao = self.ctx.vertex_array(
            prog,
            [(self._buffer, "3f 3f 1f", "in_pos", "in_color", "in_bright")],
        )

    def build_line(self, state, slice_x: float) -> int:
        """
        Build line strip: for each layer, sample activity at slice_x (column index),
        plot as (slice_x, y, z) where z is the grid row and y is layer-dependent.
        Actually: 1D profile = activity at that X for each Z (row). So we get a curve
        (slice_x, layer_y, z) with vertices for each z, and the "height" in the graph
        could be the activity value. Simpler: draw a horizontal line per layer at the
        layer's Y, with segments (slice_x, y, z) for z in [-0.5, 0.5], and color/bright
        from activity along that row.
        """
        h, w = state.grid_shape()
        slice_col = int((slice_x + 0.5) * w)  # map slice_x [-0.5,0.5] to col [0,w-1]
        slice_col = max(0, min(w - 1, slice_col))
        layout = signal_flow_slab_layout()
        verts = []
        x_pos = slice_x
        mapping = {
            "Stimulus": np.sum(state.stimulus_spectrum, axis=-1) if state.stimulus_spectrum is not None else np.zeros((h, w)),
            "Cones": state.cone_L if state.cone_L is not None else np.zeros((h, w)),
            "Horizontal": state.h_activation if state.h_activation is not None else np.zeros((h, w)),
            "Bipolar": state.bp_diffuse_on if state.bp_diffuse_on is not None else np.zeros((h, w)),
            "Amacrine": state.amacrine_aii if state.amacrine_aii is not None else np.zeros((h, w)),
            "RGC": state.fr_midget_on_L if state.fr_midget_on_L is not None else np.zeros((h, w)),
        }
        colors = [
            (0.9, 0.9, 0.5), (1.0, 0.5, 0.2), (0.9, 0.9, 0.3),
            (1.0, 0.6, 0.0), (0.7, 0.3, 0.9), (1.0, 0.2, 0.2),
        ]
        for idx, (name, y_top, thick) in enumerate(layout):
            grid = mapping.get(name)
            if grid is None:
                continue
            row = grid[:, slice_col]
            g_max = max(1e-9, float(np.max(grid)))
            col = colors[idx % len(colors)]
            y_c = y_top - thick / 2
            height_scale = 0.3  # activity modulates Y offset (oscilloscope trace)
            for i in range(h - 1):
                z0 = float((i - h / 2 + 0.5) / max(h, 1))
                z1 = float((i + 1 - h / 2 + 0.5) / max(h, 1))
                b0 = max(0, min(1, float(row[i]) / g_max))
                b1 = max(0, min(1, float(row[i + 1]) / g_max))
                y0 = y_c + b0 * height_scale
                y1 = y_c + b1 * height_scale
                verts.extend([x_pos, y0, z0, col[0], col[1], col[2], 0.5 + 0.5 * b0])
                verts.extend([x_pos, y1, z1, col[0], col[1], col[2], 0.5 + 0.5 * b1])
        if not verts:
            return 0
        arr = np.array(verts, dtype=np.float32)
        self._buffer.write(arr.tobytes())
        return len(verts) // 7

    def draw(self, state, mvp: np.ndarray, slice_x: float) -> None:
        self._ensure_resources()
        n = self.build_line(state, slice_x)
        if n < 2:
            return
        self._vao.program["u_mvp"].write(mvp.T.tobytes())
        self._vao.render(moderngl.LINES, vertices=n)
