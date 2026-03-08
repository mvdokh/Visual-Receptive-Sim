"""
Per-layer activity heatmap strips: thin vertical rectangles to the right of each slab.
Each strip shows a rolling (slab_height_px × T) texture with firing colormap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import moderngl
import numpy as np

from src.config import signal_flow_slab_layout

TRACE_TIME_STEPS = 64
STRIP_WIDTH_WORLD = 0.14  # wide enough to read oscilloscope
STRIP_X_OFFSET = 0.65
TRACE_LINE_AMPLITUDE = 0.5  # how far the trace line wiggles (fraction of strip width)


@dataclass
class LayerTraceStrips:
    """
    Renders one thin vertical strip per slab, to the right of the slab (X=+0.65).
    Each strip texture: (slab_height_px, T) rolling buffer; colormap = firing.
    """

    ctx: moderngl.Context
    slab_layout: list  # list of (name, y_top, thickness)
    slab_height_px: int = 64  # downsample 1D profile to this height per strip

    _vaos: list = field(default_factory=list, init=False)
    _textures: list = field(default_factory=list, init=False)
    _program: Optional[moderngl.Program] = field(default=None, init=False)
    _line_vao: Optional[moderngl.VertexArray] = field(default=None, init=False)
    _line_buffer: Optional[moderngl.Buffer] = field(default=None, init=False)
    _line_program: Optional[moderngl.Program] = field(default=None, init=False)

    def _ensure_resources(self) -> None:
        if self._program is not None:
            return
        self._program = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_uv;
                uniform mat4 u_mvp;
                uniform vec3 u_center;
                uniform float u_half_w;
                uniform float u_half_h;
                out vec2 v_uv;
                void main() {
                    vec3 pos = u_center + vec3(
                        (in_uv.x * 2.0 - 1.0) * u_half_w,
                        (in_uv.y * 2.0 - 1.0) * u_half_h,
                        0.0
                    );
                    gl_Position = u_mvp * vec4(pos, 1.0);
                    v_uv = in_uv;
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D u_tex;
                uniform float u_opacity;
                in vec2 v_uv;
                out vec4 fragColor;
                void main() {
                    vec4 c = texture(u_tex, v_uv);
                    fragColor = vec4(c.rgb, c.a * u_opacity);
                }
            """,
        )
        # Fullscreen quad for each strip: 2 triangles
        quad = np.array([
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 1.0, 0.0, 1.0,
        ], dtype=np.float32)
        vbo = self.ctx.buffer(quad.tobytes())
        for _ in self.slab_layout:
            self._vaos.append(
                self.ctx.vertex_array(
                    self._program,
                    [(vbo, "2f", "in_uv")],
                )
            )
            self._textures.append(None)
        # Oscilloscope trace line: pos (3) + bright (1) per vertex
        self._line_program = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos;
                in float in_bright;
                uniform mat4 u_mvp;
                out float v_bright;
                void main() {
                    gl_Position = u_mvp * vec4(in_pos, 1.0);
                    v_bright = in_bright;
                }
            """,
            fragment_shader="""
                #version 330
                in float v_bright;
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(1.0, 1.0, 0.85, 0.95 * v_bright);
                }
            """,
        )
        self._line_buffer = self.ctx.buffer(reserve=8 * 1024)  # enough for 6 layers * ~70 verts * 4 floats
        self._line_vao = self.ctx.vertex_array(
            self._line_program,
            [(self._line_buffer, "3f 1f", "in_pos", "in_bright")],
        )

    def _ensure_texture(self, idx: int, h: int, t: int) -> None:
        if self._textures[idx] is not None and self._textures[idx].size == (t, h):
            return
        if self._textures[idx] is not None:
            self._textures[idx].release()
        self._textures[idx] = self.ctx.texture((t, h), 4, data=np.zeros((h, t, 4), dtype=np.uint8).tobytes())

    def update_buffers(
        self,
        state,
        slice_x: float,
        buffers: np.ndarray,
    ) -> None:
        """
        Update rolling buffers: shift left by 1 column, write current 1D profile at slice_x
        into rightmost column for each layer. Then upload to GPU textures (firing colormap).
        buffers: (num_layers, slab_height_px, T) float32, modified in place.
        """
        h, w = state.grid_shape()
        slice_col = int((slice_x + 0.5) * w)
        slice_col = max(0, min(w - 1, slice_col))
        layout = self.slab_layout
        mapping = {
            "Stimulus": np.sum(state.stimulus_spectrum, axis=-1) if state.stimulus_spectrum is not None else np.zeros((h, w)),
            "Cones": state.cone_L if state.cone_L is not None else np.zeros((h, w)),
            "Horizontal": state.h_activation if state.h_activation is not None else np.zeros((h, w)),
            "Bipolar": state.bp_diffuse_on if state.bp_diffuse_on is not None else np.zeros((h, w)),
            "Amacrine": state.amacrine_aii if state.amacrine_aii is not None else np.zeros((h, w)),
            "RGC": state.fr_midget_on_L if state.fr_midget_on_L is not None else np.zeros((h, w)),
        }
        H = self.slab_height_px
        T = TRACE_TIME_STEPS
        self._ensure_resources()
        for idx, (name, _y_top, _thick) in enumerate(layout):
            if idx >= buffers.shape[0]:
                break
            grid = mapping.get(name)
            if grid is None:
                continue
            # 1D profile at slice_x (column slice_col), along Y (rows), downsampled to H
            row = grid[:, slice_col].astype(np.float32)
            if row.size == 0:
                continue
            indices = np.linspace(0, row.size - 1, H, dtype=np.intp)
            profile = row[indices]
            # Rolling: shift left, write new column on the right
            buf = buffers[idx]
            buf[:, :-1] = buf[:, 1:]
            buf[:, -1] = profile
            # Firing colormap: black -> amber -> white
            g = buf.copy()
            g_max = max(1e-9, float(np.max(g)))
            n = g / g_max
            rgba = np.zeros((H, T, 4), dtype=np.float32)
            rgba[..., 0] = np.minimum(1.0, n * 2.0)
            rgba[..., 1] = np.minimum(1.0, n * 1.2)
            rgba[..., 2] = np.minimum(1.0, n * 0.5)
            rgba[..., 3] = n
            tex_data = (np.clip(rgba, 0, 1) * 255.0).astype(np.uint8)
            self._ensure_texture(idx, H, T)
            self._textures[idx].write(tex_data.tobytes())

    def draw(
        self,
        mvp: np.ndarray,
        buffers: np.ndarray,
        fog_near: float = 0.3,
        fog_far: float = 0.95,
    ) -> None:
        """Draw all strips (heatmap texture) then oscilloscope trace line on each."""
        self._ensure_resources()
        layout = self.slab_layout
        half_w = STRIP_WIDTH_WORLD / 2.0
        H = self.slab_height_px
        for idx, (name, y_top, thick) in enumerate(layout):
            if idx >= len(self._textures) or self._textures[idx] is None:
                continue
            y_center = y_top - thick / 2.0
            center = np.array([STRIP_X_OFFSET, y_center, 0.0], dtype=np.float32)
            half_h = thick / 2.0
            self._program["u_mvp"].write(mvp.T.tobytes())
            self._program["u_center"].write(center.tobytes())
            self._program["u_half_w"].value = half_w
            self._program["u_half_h"].value = half_h
            self._program["u_tex"].value = 0
            self._program["u_opacity"].value = 0.9
            self._textures[idx].use(location=0)
            self._vaos[idx].render(moderngl.TRIANGLES, vertices=6)
            # Oscilloscope trace line: horizontal = space (sweep), vertical = activity (deflection)
            if idx < buffers.shape[0]:
                profile = buffers[idx][:, -1]
                g_max = max(1e-9, float(np.max(profile)))
                n = np.clip(profile / g_max, 0.0, 1.0)
                x_left = STRIP_X_OFFSET - half_w
                verts = []
                for i in range(H):
                    frac = i / max(H - 1, 1)
                    x_pt = x_left + frac * (2.0 * half_w)  # horizontal = space along slab
                    y_pt = y_center + (float(n[i]) - 0.5) * thick * TRACE_LINE_AMPLITUDE  # vertical = activity
                    verts.extend([x_pt, y_pt, 0.0, 0.3 + 0.7 * float(n[i])])
                if verts:
                    arr = np.array(verts, dtype=np.float32)
                    self._line_buffer.write(arr.tobytes())
                    self._line_program["u_mvp"].write(mvp.T.tobytes())
                    self._line_vao.render(moderngl.LINE_STRIP, vertices=H)


def allocate_trace_buffers(num_layers: int, slab_height_px: int = 64) -> np.ndarray:
    """Allocate (num_layers, slab_height_px, T) float32 zeros for rolling buffers."""
    return np.zeros((num_layers, slab_height_px, TRACE_TIME_STEPS), dtype=np.float32)
