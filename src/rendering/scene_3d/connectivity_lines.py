"""
Connectivity columns: thin vertical lines connecting cone -> bipolar -> RGC.
Brightness = activity product. Subsample every 8th cell (~32x32).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import moderngl
import numpy as np


@dataclass
class ConnectivityLines:
    """Draws vertical line segments through the layer stack."""

    ctx: moderngl.Context
    slab_layout: list  # list of (name, y_top, thickness)
    subsample: int = 8  # one column per N cells

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
                    fragColor = vec4(v_color * v_bright, 0.85);
                }
            """,
        )
        # Format: pos (3) + color (3) + bright (1) = 7 floats per vertex, 2 verts per segment
        self._buffer = self.ctx.buffer(reserve=256 * 256 * 7 * 4 * 2)  # max lines
        self._vao = self.ctx.vertex_array(
            prog,
            [(self._buffer, "3f 3f 1f", "in_pos", "in_color", "in_bright")],
        )

    def build_segments(self, state) -> int:
        """
        Build line segments: cone(x,z) -> bipolar(x,z) -> rgc(x,z).
        Returns number of segments (each segment = 2 vertices).
        """
        h, w = state.grid_shape()
        step = self.subsample
        ny, nx = max(1, h // step), max(1, w // step)
        field = 1.0
        xs = (np.arange(nx) - nx / 2 + 0.5) * (field / max(nx, 1))
        zs = (np.arange(ny) - ny / 2 + 0.5) * (field / max(ny, 1))

        # Layer centers and grids
        cone_L = state.cone_L if state.cone_L is not None else np.zeros((h, w))
        bp = state.bp_diffuse_on if state.bp_diffuse_on is not None else np.zeros((h, w))
        rgc = state.fr_midget_on_L if state.fr_midget_on_L is not None else np.zeros((h, w))
        g_max_c = max(1e-9, float(np.max(cone_L)))
        g_max_b = max(1e-9, float(np.max(bp)))
        g_max_r = max(1e-9, float(np.max(rgc)))

        # Slab Y positions (center of each layer)
        y_cone = 4.7   # cones center
        y_bp = 3.58   # bipolar center
        y_rgc = 2.85  # RGC center

        # Colors: excitatory warm amber, horizontal/inhibitory cool blue
        color_excit = (1.0, 0.75, 0.2)   # amber
        color_inhib = (0.3, 0.5, 1.0)    # cool blue

        verts = []
        for i in range(ny):
            for j in range(nx):
                pi, pj = min(i * step, h - 1), min(j * step, w - 1)
                x, z = float(xs[j]), float(zs[i])
                act_c = float(cone_L[pi, pj]) / g_max_c
                act_b = float(bp[pi, pj]) / g_max_b
                act_r = float(rgc[pi, pj]) / g_max_r
                bright = act_c * act_b * 0.5 + act_b * act_r * 0.5
                if bright < 0.05:
                    continue
                bright = min(1.0, bright)
                col = color_excit
                # Cone -> Bipolar
                verts.extend([x, y_cone, z, col[0], col[1], col[2], bright])
                verts.extend([x, y_bp, z, col[0], col[1], col[2], bright])
                # Bipolar -> RGC
                verts.extend([x, y_bp, z, col[0], col[1], col[2], bright])
                verts.extend([x, y_rgc, z, col[0], col[1], col[2], bright])
        if not verts:
            return 0
        arr = np.array(verts, dtype=np.float32)
        self._buffer.write(arr.tobytes())
        return len(verts) // 7  # vertex count

    def draw(self, state, mvp: np.ndarray) -> None:
        self._ensure_resources()
        n_verts = self.build_segments(state)
        if n_verts < 2:
            return
        self._vao.program["u_mvp"].write(mvp.T.tobytes())
        self._vao.render(moderngl.LINES, vertices=n_verts)
