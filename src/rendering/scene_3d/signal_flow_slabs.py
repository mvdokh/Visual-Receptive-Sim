"""
Signal Flow Column: thick horizontal slabs arranged top-to-bottom.
Each slab has a top face (XY spatial heatmap) and front face (XZ profile).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import moderngl
import numpy as np

from src.config import signal_flow_slab_layout
from src.rendering.heatmap import ColormapName, grid_to_rgba


@dataclass
class LayerSlab:
    """
    Thick horizontal slab with top face (spatial heatmap) and front face (activity profile).
    Coordinate: X = spatial cols [-0.5,0.5], Y = layer depth (vertical), Z = spatial rows [-0.5,0.5].
    """

    ctx: moderngl.Context
    label: str
    y_top: float
    thickness: float
    grid: np.ndarray
    colormap: ColormapName = "firing"
    opacity: float = 0.85
    visible: bool = True

    _texture: Optional[moderngl.Texture] = field(default=None, init=False)
    _vao_slab: Optional[moderngl.VertexArray] = field(default=None, init=False)

    def ensure_resources(self) -> None:
        if self._vao_slab is not None:
            return
        if self.grid.ndim == 3 and self.grid.shape[2] == 4:
            rgba = np.ascontiguousarray(self.grid.astype(np.float32))
            h, w = rgba.shape[:2]
        else:
            h, w = self.grid.shape
            rgba = grid_to_rgba(self.grid, self.colormap)
        tex_data = (np.clip(rgba, 0, 1) * 255.0).astype("u1")
        self._texture = self.ctx.texture((w, h), 4, data=tex_data.tobytes())
        self._texture.build_mipmaps()

        # Box geometry: 6 faces. We draw top (XZ) and front (XY) only for clarity.
        # Vertices: 8 corners of box. X [-0.5,0.5], Y [y_bot, y_top], Z [-0.5,0.5]
        y_bot = self.y_top - self.thickness
        # Top face: 4 verts, Y=y_top, XZ plane
        # Front face: 4 verts, Z=0.5, XY plane
        # Use triangle list for slab box (simplified: top + front quads)
        verts = np.array([
            # Top face (Y = y_top): 2 triangles
            -0.5, self.y_top, -0.5,  0.0, 0.0,
             0.5, self.y_top, -0.5,  1.0, 0.0,
            -0.5, self.y_top,  0.5,  0.0, 1.0,
             0.5, self.y_top, -0.5,  1.0, 0.0,
             0.5, self.y_top,  0.5,  1.0, 1.0,
            -0.5, self.y_top,  0.5,  0.0, 1.0,
            # Front face (Z = 0.5): X,Y,U,V (U = (X+0.5), V = (Y-y_bot)/thickness)
            -0.5, y_bot, 0.5,  0.0, 0.0,
             0.5, y_bot, 0.5,  1.0, 0.0,
            -0.5, self.y_top, 0.5,  0.0, 1.0,
             0.5, y_bot, 0.5,  1.0, 0.0,
             0.5, self.y_top, 0.5,  1.0, 1.0,
            -0.5, self.y_top, 0.5,  0.0, 1.0,
        ], dtype="f4")
        vbo = self.ctx.buffer(verts.tobytes())
        prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos;
                in vec2 in_uv;
                uniform mat4 u_mvp;
                out vec2 v_uv;
                out float v_depth;
                void main() {
                    v_uv = in_uv;
                    vec4 clip = u_mvp * vec4(in_pos, 1.0);
                    v_depth = clip.z / clip.w;
                    gl_Position = clip;
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D u_tex;
                uniform float u_opacity;
                uniform float u_fog_near;
                uniform float u_fog_far;
                in vec2 v_uv;
                in float v_depth;
                out vec4 fragColor;
                void main() {
                    vec4 c = texture(u_tex, v_uv);
                    float fog = clamp((v_depth - u_fog_near) / (u_fog_far - u_fog_near), 0.0, 1.0);
                    vec3 fogColor = vec3(0.05, 0.05, 0.12);
                    vec3 rgb = mix(c.rgb, fogColor, fog);
                    fragColor = vec4(rgb, c.a * u_opacity);
                }
            """,
        )
        self._vao_slab = self.ctx.vertex_array(prog, [(vbo, "3f 2f", "in_pos", "in_uv")])

    def update_from_grid(self, grid: np.ndarray) -> None:
        if grid.ndim == 3 and grid.shape[2] == 4:
            rgba = np.ascontiguousarray(grid.astype(np.float32))
        else:
            self.grid = grid.astype(np.float32)
            rgba = grid_to_rgba(self.grid, self.colormap)
        if self._texture is None:
            return
        h, w = rgba.shape[:2]
        tex_data = (np.clip(rgba, 0, 1) * 255.0).astype("u1")
        if self._texture.size != (w, h):
            self._texture.release()
            self._texture = self.ctx.texture((w, h), 4, data=tex_data.tobytes())
        else:
            self._texture.write(tex_data.tobytes())

    def draw(self, mvp: np.ndarray, fog_near: float = 0.3, fog_far: float = 0.95) -> None:
        if not self.visible:
            return
        self.ensure_resources()
        assert self._vao_slab is not None
        p = self._vao_slab.program
        p["u_mvp"].write(mvp.T.tobytes())
        p["u_opacity"].value = self.opacity
        p["u_fog_near"].value = fog_near
        p["u_fog_far"].value = fog_far
        p["u_tex"].value = 0
        self._texture.use(location=0)
        self._vao_slab.render(moderngl.TRIANGLES, vertices=12)  # 6 verts x 2 faces


def create_slabs(ctx: moderngl.Context, state) -> Dict[str, LayerSlab]:
    """Create LayerSlab dict from layout and state."""
    from src.rendering.heatmap import spectrum_to_stimulus_rgba

    slabs = {}
    cfg = state.config
    wl = cfg.spectral.wavelengths if cfg else np.arange(380, 701, 5, dtype=np.float32)
    layout = signal_flow_slab_layout()
    mapping = {
        "Stimulus": (spectrum_to_stimulus_rgba(state.stimulus_spectrum, wl)
            if state.stimulus_spectrum is not None
            else np.zeros((*state.grid_shape(), 4), dtype=np.float32)),
        "Cones": state.cone_L,
        "Horizontal": state.h_activation,
        "Bipolar": state.bp_diffuse_on,
        "Amacrine": state.amacrine_aii,
        "RGC": state.fr_midget_on_L,
    }
    for name, y_top, thick in layout:
        grid = mapping.get(name)
        if grid is None:
            grid = np.zeros(state.grid_shape(), dtype=np.float32)
        slab = LayerSlab(
            ctx=ctx, label=name, y_top=y_top, thickness=thick,
            grid=grid, colormap="firing", opacity=0.85,
        )
        slabs[name] = slab
    return slabs
