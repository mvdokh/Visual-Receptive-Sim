"""
ModernGL rendering context for the 3D Signal Flow Column.

Scene: thick horizontal slabs, connectivity lines (four types), cell spheres with
GPU additive bloom, per-layer trace strips (rolling heatmaps), box-blur post-pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import moderngl
import numpy as np

from src.config import GlobalConfig, signal_flow_slab_layout
from src.rendering.heatmap import spectrum_to_stimulus_rgba
from src.rendering.scene_3d.camera import OrbitCamera
from src.rendering.scene_3d.cell_spheres import CellSpheresRenderer
from src.rendering.scene_3d.connectivity_lines import ConnectivityLines
from src.rendering.scene_3d.layer_trace_strips import (
    LayerTraceStrips,
    allocate_trace_buffers,
)
from src.rendering.scene_3d.signal_flow_slabs import LayerSlab, create_slabs
from src.simulation.state import SimState


@dataclass
class RenderContext:
    """Signal Flow Column: slabs, connectivity, cells, layer trace strips, bloom, blur."""

    size: Tuple[int, int] = (1024, 1024)
    config: GlobalConfig | None = None

    ctx: moderngl.Context = field(init=False)
    fbo_msaa: Optional[moderngl.Framebuffer] = field(default=None, init=False)
    fbo_resolve: Optional[moderngl.Framebuffer] = field(default=None, init=False)
    fbo_blur: Optional[moderngl.Framebuffer] = field(default=None, init=False)
    _blur_vao: Optional[moderngl.VertexArray] = field(default=None, init=False)
    _blur_program: Optional[moderngl.Program] = field(default=None, init=False)

    slabs: Dict[str, LayerSlab] = field(default_factory=dict, init=False)
    camera: OrbitCamera = field(default_factory=OrbitCamera, init=False)
    cell_spheres: Optional[CellSpheresRenderer] = field(default=None, init=False)
    connectivity: Optional[ConnectivityLines] = field(default=None, init=False)
    layer_trace_strips: Optional[LayerTraceStrips] = field(default=None, init=False)
    trace_buffers: Optional[np.ndarray] = field(default=None, init=False)  # (num_layers, slab_height_px, T)

    show_cells: bool = True
    show_connectivity: bool = True
    show_cone_to_horizontal: bool = True
    show_cone_to_bipolar: bool = True
    show_bipolar_to_amacrine: bool = True
    show_bipolar_to_rgc: bool = True
    slice_x: float = 0.0
    connectivity_dirty: bool = False
    _slab_layout: list = field(default_factory=list, init=False)
    _slab_dirty_key: Dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.ctx = moderngl.create_standalone_context()
        self._resize(*self.size)

    def _resize(self, width: int, height: int) -> None:
        # Single-sample FBO (no MSAA); second FBO for box-blur post-pass
        color_tex = self.ctx.texture((width, height), 4)
        # Slight filtering to soften jagged edges without heavy post-processing
        color_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        depth_rb = self.ctx.depth_renderbuffer((width, height))
        self.fbo_msaa = self.ctx.framebuffer(
            color_attachments=[color_tex],
            depth_attachment=depth_rb,
        )
        self.fbo_resolve = self.fbo_msaa
        blur_tex = self.ctx.texture((width, height), 4)
        self.fbo_blur = self.ctx.framebuffer(color_attachments=[blur_tex])
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

    @property
    def color_texture(self):
        return self.fbo_resolve.color_attachments[0]

    def ensure_scene(self, state: SimState) -> None:
        if self.slabs:
            return
        self._slab_layout = list(signal_flow_slab_layout())
        # Create slabs once; their textures will be updated each frame.
        self.slabs = create_slabs(self.ctx, state)
        self.connectivity = ConnectivityLines(self.ctx, self._slab_layout, subsample=8)
        slab_height_px = 64
        self.layer_trace_strips = LayerTraceStrips(
            self.ctx, self._slab_layout, slab_height_px=slab_height_px
        )
        self.trace_buffers = allocate_trace_buffers(len(self._slab_layout), slab_height_px)
        subsample = getattr(state.config, "cell_subsample", 8) if state.config else 8
        self.cell_spheres = CellSpheresRenderer(self.ctx, subsample=subsample)

    def update_from_state(self, state: SimState) -> None:
        """
        Refresh 3D scene objects from the latest simulation state.

        To ensure the 3D slabs (especially the Stimulus slab) never get stuck
        on an old texture, we rebuild the slab set from scratch each frame
        using `create_slabs`, then restore per-layer visibility/opacity.
        This guarantees the Stimulus slab matches the current stimulus just
        like the 2D Stimulus heatmap.
        """
        # Preserve current visibility / opacity so UI toggles persist.
        prev_settings: Dict[str, Tuple[bool, float]] = {}
        for label, slab in self.slabs.items():
            prev_settings[label] = (slab.visible, slab.opacity)

        # Recreate all slabs from the *current* state.
        self.slabs = create_slabs(self.ctx, state)

        # Reapply visibility / opacity where possible.
        for label, slab in self.slabs.items():
            if label in prev_settings:
                vis, op = prev_settings[label]
                slab.visible = vis
                slab.opacity = op

        if self.layer_trace_strips is not None and self.trace_buffers is not None:
            self.layer_trace_strips.update_buffers(state, self.slice_x, self.trace_buffers)

    def _ensure_blur_resources(self, width: int, height: int) -> None:
        if self._blur_program is not None:
            return
        self._blur_program = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_uv;
                out vec2 v_uv;
                void main() {
                    v_uv = in_uv;
                    gl_Position = vec4(in_uv * 2.0 - 1.0, 0.0, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D u_tex;
                uniform vec2 u_texel_size;
                in vec2 v_uv;
                out vec4 fragColor;
                void main() {
                    vec4 sum = texture(u_tex, v_uv);
                    sum += texture(u_tex, v_uv + vec2(-1,-1) * u_texel_size);
                    sum += texture(u_tex, v_uv + vec2( 0,-1) * u_texel_size);
                    sum += texture(u_tex, v_uv + vec2( 1,-1) * u_texel_size);
                    sum += texture(u_tex, v_uv + vec2(-1, 0) * u_texel_size);
                    sum += texture(u_tex, v_uv + vec2( 1, 0) * u_texel_size);
                    sum += texture(u_tex, v_uv + vec2(-1, 1) * u_texel_size);
                    sum += texture(u_tex, v_uv + vec2( 0, 1) * u_texel_size);
                    sum += texture(u_tex, v_uv + vec2( 1, 1) * u_texel_size);
                    fragColor = sum / 9.0;
                }
            """,
        )
        quad = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float32)
        vbo = self.ctx.buffer(quad.tobytes())
        self._blur_vao = self.ctx.vertex_array(
            self._blur_program,
            [(vbo, "2f", "in_uv")],
        )

    def _blur_pass(self, width: int, height: int) -> None:
        """3×3 box blur: render from fbo_msaa to fbo_blur."""
        self._ensure_blur_resources(width, height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.fbo_blur.use()
        self.fbo_msaa.color_attachments[0].use(location=0)
        self._blur_program["u_tex"].value = 0
        self._blur_program["u_texel_size"].value = (1.0 / width, 1.0 / height)
        self._blur_vao.render(moderngl.TRIANGLES, vertices=6)
        self.ctx.enable(moderngl.DEPTH_TEST)

    def _read_fbo_to_uint8(self, width: int, height: int, from_blur: bool = False) -> np.ndarray:
        """Read FBO to uint8 RGBA (flip Y for display). from_blur=True reads fbo_blur."""
        fbo = self.fbo_blur if from_blur and self.fbo_blur else self.fbo_msaa
        fbo.use()
        data = fbo.color_attachments[0].read()
        img = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))
        return np.flipud(img)

    def render_3d(self, state: SimState) -> np.ndarray:
        """Render Signal Flow Column, resolve MSAA, apply bloom, return uint8 RGBA."""
        self.fbo_msaa.use()
        # Modern dark background with subtle blue tint
        self.ctx.clear(0.02, 0.02, 0.06, 1.0)
        self.update_from_state(state)

        width, height = self.fbo_msaa.color_attachments[0].size
        aspect = width / max(height, 1)
        field_size = self.config.retina.field_size_deg if self.config else 1.0
        model = np.eye(4, dtype=np.float32)
        mvp = self.camera.mvp(model, aspect)
        fog_near, fog_far = 0.3, 0.95

        # 1. Slabs (back to front)
        for slab in self.slabs.values():
            slab.draw(mvp, fog_near, fog_far)

        # 2. Connectivity lines (four types, toggles + weights)
        if self.show_connectivity and self.connectivity is not None:
            cw = getattr(state.config, "connectivity_weights", None) if state.config else None
            show_ch = getattr(self, "show_cone_to_horizontal", True)
            show_cb = getattr(self, "show_cone_to_bipolar", True)
            show_ba = getattr(self, "show_bipolar_to_amacrine", True)
            show_br = getattr(self, "show_bipolar_to_rgc", True)
            self.connectivity.draw(
                state, mvp, self.slice_x, self.connectivity_dirty,
                show_cone_to_horizontal=show_ch,
                show_cone_to_bipolar=show_cb,
                show_bipolar_to_amacrine=show_ba,
                show_bipolar_to_rgc=show_br,
                weights=cw,
            )
            self.connectivity_dirty = False

        # 3. Cell spheres (solid)
        if self.show_cells and self.cell_spheres is not None:
            self.cell_spheres.draw(state, mvp, field_size)

        # 4. Cell spheres additive bloom (1.8× radius, 0.12 alpha, GPU only)
        if self.show_cells and self.cell_spheres is not None:
            self.cell_spheres.draw_bloom(state, mvp, field_size)

        # 5. Per-layer trace strips (right of each slab)
        if self.layer_trace_strips is not None and self.trace_buffers is not None:
            self.layer_trace_strips.draw(mvp, self.trace_buffers, fog_near, fog_far)

        # 6. Read back color buffer (no extra blur; sharper image)
        img = self._read_fbo_to_uint8(width, height, from_blur=False)
        return img
