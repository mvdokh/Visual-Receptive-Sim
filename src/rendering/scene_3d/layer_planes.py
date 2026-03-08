from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import moderngl
import numpy as np

from src.rendering.heatmap import ColormapName, grid_to_rgba


@dataclass
class LayerPlane:
    """
    Flat quad mesh rendered as a heatmap texture at a fixed Z.
    """

    ctx: moderngl.Context
    z_position: float
    label: str
    grid: np.ndarray  # (H, W) float32
    colormap: ColormapName = "firing"
    opacity: float = 0.6
    visible: bool = True

    _texture: Optional[moderngl.Texture] = field(default=None, init=False, repr=False)
    _vao: Optional[moderngl.VertexArray] = field(default=None, init=False, repr=False)
    _vao_3d: Optional[moderngl.VertexArray] = field(default=None, init=False, repr=False)

    def ensure_gpu_resources(self) -> None:
        """Create the quad geometry and texture if they do not exist yet."""
        if self._vao is not None and self._texture is not None:
            return

        h, w = self.grid.shape
        # Simple fullscreen quad in X/Y; Z applied in shader as uniform
        vertices = np.array(
            [
                -1.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                -1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype="f4",
        )
        vbo = self.ctx.buffer(vertices.tobytes())

        prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                in vec2 in_uv;
                out vec2 v_uv;
                uniform float u_z;
                void main() {
                    v_uv = in_uv;
                    // For now we render each layer as a fullscreen quad in 2D.
                    gl_Position = vec4(in_pos.xy, 0.0, 1.0);
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

        self._vao = self.ctx.vertex_array(
            prog,
            [
                (vbo, "2f 2f", "in_pos", "in_uv"),
            ],
        )

        rgba = grid_to_rgba(self.grid, self.colormap)
        # ModernGL's default texture dtype expects 1 byte per component.
        tex_data = (rgba * 255.0).astype("u1")
        self._texture = self.ctx.texture((w, h), 4, data=tex_data.tobytes())
        self._texture.build_mipmaps()

    def update_from_grid(self, grid: np.ndarray) -> None:
        """Update the backing texture from a new activation grid."""
        self.grid = grid.astype(np.float32)
        if self._texture is None:
            return
        h, w = self.grid.shape
        rgba = grid_to_rgba(self.grid, self.colormap)
        tex_data = (rgba * 255.0).astype("u1")
        if self._texture.size != (w, h):
            self._texture.release()
            self._texture = self.ctx.texture((w, h), 4, data=tex_data.tobytes())
        else:
            self._texture.write(tex_data.tobytes())

    def _ensure_3d_resources(self) -> None:
        if self._vao_3d is not None:
            return
        z = self.z_position
        # World-space quad: (-0.5,-0.5,z) to (0.5,0.5,z), UV 0-1
        vertices = np.array([
            -0.5, -0.5, z, 0.0, 0.0,
             0.5, -0.5, z, 1.0, 0.0,
            -0.5,  0.5, z, 0.0, 1.0,
             0.5,  0.5, z, 1.0, 1.0,
        ], dtype="f4")
        vbo = self.ctx.buffer(vertices.tobytes())
        prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos;
                in vec2 in_uv;
                uniform mat4 u_mvp;
                out vec2 v_uv;
                void main() {
                    v_uv = in_uv;
                    gl_Position = u_mvp * vec4(in_pos, 1.0);
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
        self._vao_3d = self.ctx.vertex_array(prog, [(vbo, "3f 2f", "in_pos", "in_uv")])

    def draw(self) -> None:
        """2D fullscreen draw (for 2D heatmap mode)."""
        if not self.visible:
            return
        self.ensure_gpu_resources()
        assert self._vao is not None
        prog = self._vao.program
        prog["u_opacity"].value = self.opacity
        self._texture.use(location=0)
        self._vao.render(mode=moderngl.TRIANGLE_STRIP)

    def draw_3d(self, mvp: np.ndarray) -> None:
        """3D draw with MVP matrix (for 3D stack view)."""
        if not self.visible:
            return
        self.ensure_gpu_resources()
        self._ensure_3d_resources()
        assert self._vao_3d is not None
        self._vao_3d.program["u_mvp"].write(mvp.T.tobytes())
        self._vao_3d.program["u_opacity"].value = self.opacity
        self._vao_3d.program["u_tex"].value = 0
        self._texture.use(location=0)
        self._vao_3d.render(mode=moderngl.TRIANGLE_STRIP)

