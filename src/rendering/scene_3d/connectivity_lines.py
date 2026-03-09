"""
Connectivity columns: vertical lines for cone→horizontal, cone→bipolar, bipolar→amacrine, bipolar→RGC.
Each type togglable; brightness = source_activity * target_activity * type_weight.
Fan-in ratios (photoreceptor→RGC ~100:1, fovea ~1:1 vs periphery up to 30:1) from
src.simulation.bio_constants; use fovea/periphery toggle in UI to switch (CONE_RGC_RATIO_FOVEA vs CONE_RGC_RATIO_PERIPHERY).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import moderngl
import numpy as np


def _y_center(slab_layout: list, name: str) -> float:
    for n, y_top, thick in slab_layout:
        if n == name:
            return y_top - thick / 2.0
    return 3.0


@dataclass
class ConnectivityLines:
    """Draws vertical line segments for four connection types (togglable, weight-scaled)."""

    ctx: moderngl.Context
    slab_layout: list  # list of (name, y_top, thickness)
    subsample: int = 8
    max_segments: int = 512

    _vao: Optional[moderngl.VertexArray] = field(default=None, init=False)
    _buffer: Optional[moderngl.Buffer] = field(default=None, init=False)
    _last_slice_x: Optional[float] = field(default=None, init=False)
    _last_weights_hash: int = field(default=0, init=False)
    _cached_vertex_count: int = field(default=0, init=False)

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

    def build_segments(
        self,
        state: Any,
        show_cone_to_horizontal: bool = True,
        show_cone_to_bipolar: bool = True,
        show_bipolar_to_amacrine: bool = True,
        show_bipolar_to_rgc: bool = True,
        weights: Optional[Any] = None,
    ) -> int:
        """Build line segments for enabled types. weights: object with cone_to_horizontal, etc."""
        h, w = state.grid_shape()
        step = self.subsample
        ny, nx = max(1, h // step), max(1, w // step)
        field = 1.0
        xs = (np.arange(nx) - nx / 2 + 0.5) * (field / max(nx, 1))
        zs = (np.arange(ny) - ny / 2 + 0.5) * (field / max(ny, 1))

        cone_L = state.cone_L if state.cone_L is not None else np.zeros((h, w))
        h_act = state.h_activation if state.h_activation is not None else np.zeros((h, w))
        bp = state.bp_diffuse_on if state.bp_diffuse_on is not None else np.zeros((h, w))
        am = state.amacrine_aii if state.amacrine_aii is not None else np.zeros((h, w))
        rgc = state.fr_midget_on_L if state.fr_midget_on_L is not None else np.zeros((h, w))
        g_c = max(1e-9, float(np.max(cone_L)))
        g_h = max(1e-9, float(np.max(h_act)))
        g_b = max(1e-9, float(np.max(bp)))
        g_a = max(1e-9, float(np.max(am)))
        g_r = max(1e-9, float(np.max(rgc)))

        y_cone = _y_center(self.slab_layout, "Cones")
        y_h = _y_center(self.slab_layout, "Horizontal")
        y_bp = _y_center(self.slab_layout, "Bipolar")
        y_am = _y_center(self.slab_layout, "Amacrine")
        y_rgc = _y_center(self.slab_layout, "RGC")

        def get_weight(key: str) -> float:
            if weights is None:
                return 1.0
            return getattr(weights, key, 1.0)

        verts = []
        for i in range(ny):
            for j in range(nx):
                pi, pj = min(i * step, h - 1), min(j * step, w - 1)
                x, z = float(xs[j]), float(zs[i])
                act_c = float(cone_L[pi, pj]) / g_c
                act_h = float(h_act[pi, pj]) / g_h
                act_b = float(bp[pi, pj]) / g_b
                act_a = float(am[pi, pj]) / g_a
                act_r = float(rgc[pi, pj]) / g_r
                if show_cone_to_horizontal:
                    bright = act_c * act_h * get_weight("cone_to_horizontal")
                    if bright >= 0.05:
                        col = (0.27, 0.53, 1.0)
                        verts.extend([x, y_cone, z, col[0], col[1], col[2], min(1.0, bright)])
                        verts.extend([x, y_h, z, col[0], col[1], col[2], min(1.0, bright)])
                if show_cone_to_bipolar:
                    bright = act_c * act_b * get_weight("cone_to_bipolar")
                    if bright >= 0.05:
                        col = (1.0, 0.67, 0.13)
                        verts.extend([x, y_cone, z, col[0], col[1], col[2], min(1.0, bright)])
                        verts.extend([x, y_bp, z, col[0], col[1], col[2], min(1.0, bright)])
                if show_bipolar_to_amacrine:
                    bright = act_b * act_a * get_weight("bipolar_to_amacrine")
                    if bright >= 0.05:
                        col = (0.67, 0.27, 1.0)
                        verts.extend([x, y_bp, z, col[0], col[1], col[2], min(1.0, bright)])
                        verts.extend([x, y_am, z, col[0], col[1], col[2], min(1.0, bright)])
                if show_bipolar_to_rgc:
                    bright = act_b * act_r * get_weight("bipolar_to_rgc")
                    if bright >= 0.05:
                        col = (1.0, 0.27, 0.27)
                        verts.extend([x, y_bp, z, col[0], col[1], col[2], min(1.0, bright)])
                        verts.extend([x, y_rgc, z, col[0], col[1], col[2], min(1.0, bright)])
        if not verts:
            return 0
        n_verts = len(verts) // 7
        n_segments = n_verts // 2
        if n_segments > self.max_segments:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_segments, size=self.max_segments, replace=False)
            kept = []
            for seg_i in idx:
                for v in range(2):
                    base = (seg_i * 2 + v) * 7
                    kept.extend(verts[base : base + 7])
            verts = kept
            n_verts = len(verts) // 7
        arr = np.array(verts, dtype=np.float32)
        self._buffer.write(arr.tobytes())
        return n_verts

    def draw(
        self,
        state: Any,
        mvp: np.ndarray,
        slice_x: Optional[float] = None,
        connectivity_dirty: bool = False,
        show_cone_to_horizontal: bool = True,
        show_cone_to_bipolar: bool = True,
        show_bipolar_to_amacrine: bool = True,
        show_bipolar_to_rgc: bool = True,
        weights: Optional[Any] = None,
    ) -> None:
        self._ensure_resources()
        weights_hash = id(weights) if weights is not None else 0
        if weights is not None:
            try:
                weights_hash = hash((weights.cone_to_horizontal, weights.cone_to_bipolar, weights.horizontal_to_cone, weights.bipolar_to_amacrine, weights.amacrine_to_bipolar, weights.bipolar_to_rgc))
            except Exception:
                pass
        need_rebuild = (
            self._last_slice_x is None
            or connectivity_dirty
            or (slice_x is not None and self._last_slice_x != slice_x)
            or self._last_weights_hash != weights_hash
        )
        if need_rebuild:
            self._last_slice_x = slice_x
            self._last_weights_hash = weights_hash
            self._cached_vertex_count = self.build_segments(
                state,
                show_cone_to_horizontal=show_cone_to_horizontal,
                show_cone_to_bipolar=show_cone_to_bipolar,
                show_bipolar_to_amacrine=show_bipolar_to_amacrine,
                show_bipolar_to_rgc=show_bipolar_to_rgc,
                weights=weights,
            )
        n_verts = self._cached_vertex_count
        if n_verts < 2:
            return
        self._vao.program["u_mvp"].write(mvp.T.tobytes())
        self._vao.render(moderngl.LINES, vertices=n_verts)
