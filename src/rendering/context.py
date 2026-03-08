from __future__ import annotations

"""
ModernGL rendering context for the 3D retinal scene.

This module owns the OpenGL context, framebuffers, and high-level scene
objects (layer planes, cells, connections). For now, only layer planes
are rendered, but the structure is ready to grow into full 3D.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple

import moderngl
import numpy as np

from src.config import GlobalConfig, layer_z_positions
from src.rendering.scene_3d.camera import OrbitCamera
from src.rendering.scene_3d.cell_spheres import CellSpheresRenderer
from src.rendering.scene_3d.layer_planes import LayerPlane
from src.simulation.state import SimState


@dataclass
class RenderContext:
    """Owns the ModernGL context and scene graph."""

    size: Tuple[int, int] = (1024, 768)
    config: GlobalConfig | None = None

    ctx: moderngl.Context = field(init=False)
    fbo: moderngl.Framebuffer = field(init=False)

    layer_planes: Dict[str, LayerPlane] = field(default_factory=dict, init=False)
    camera: OrbitCamera = field(default_factory=OrbitCamera, init=False)
    cell_spheres: CellSpheresRenderer | None = field(default=None, init=False)
    show_cells: bool = True

    def __post_init__(self) -> None:
        # Standalone offscreen context; Dear PyGui will display the FBO as a texture.
        self.ctx = moderngl.create_standalone_context()
        self._resize(*self.size)

    def _resize(self, width: int, height: int) -> None:
        color_tex = self.ctx.texture((width, height), 4)
        depth_rb = self.ctx.depth_renderbuffer((width, height))
        self.fbo = self.ctx.framebuffer(color_attachments=[color_tex], depth_attachment=depth_rb)
        self.fbo.use()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

    @property
    def color_texture(self) -> moderngl.Texture:
        return self.fbo.color_attachments[0]

    def ensure_scene(self, state: SimState) -> None:
        """Create layer planes if they don't exist."""
        if self.layer_planes:
            return

        z_vals = layer_z_positions()
        grids = {
            "Stimulus": np.zeros(state.grid_shape(), dtype=np.float32),
            "Cones": state.cone_L,
            "Horizontal": state.h_activation,
            "Bipolar": state.bp_diffuse_on,
            "Amacrine": state.amacrine_aii,
            "RGC": state.fr_midget_on_L,
        }
        labels = list(grids.keys())

        for i, label in enumerate(labels):
            grid = grids[label]
            plane = LayerPlane(
                ctx=self.ctx,
                z_position=float(z_vals[min(i + 1, len(z_vals) - 1)]),
                label=label,
                grid=grid,
                colormap="firing",
                opacity=0.6,
            )
            self.layer_planes[label] = plane

    def update_from_state(self, state: SimState) -> None:
        """Propagate new activation grids into GPU textures."""
        self.ensure_scene(state)
        # Map state attributes back into named planes.
        mapping = {
            "Stimulus": np.sum(state.stimulus_spectrum, axis=-1)
            if state.stimulus_spectrum is not None
            else np.zeros(state.grid_shape(), dtype=np.float32),
            "Cones": state.cone_L,
            "Horizontal": state.h_activation,
            "Bipolar": state.bp_diffuse_on,
            "Amacrine": state.amacrine_aii,
            "RGC": state.fr_midget_on_L,
        }
        for label, grid in mapping.items():
            plane = self.layer_planes.get(label)
            if plane is not None and grid is not None:
                plane.update_from_grid(grid)

    def render(self, state: SimState) -> np.ndarray:
        """
        Render the current scene into the offscreen framebuffer and
        return an (H, W, 4) uint8 image suitable for Dear PyGui textures.
        """
        self.fbo.use()
        self.ctx.clear(0.02, 0.02, 0.04, 1.0)

        # Sync GPU-side textures with the latest simulation state.
        self.update_from_state(state)
        for plane in self.layer_planes.values():
            plane.draw()

        # Read back into system memory. RGBA, 8-bit per channel.
        width, height = self.color_texture.size
        data = self.color_texture.read()
        img = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))
        # Flip vertically because OpenGL origin is bottom-left.
        img = np.flipud(img)
        return img

    def render_3d(self, state: SimState) -> np.ndarray:
        """Render the 3D retinal stack with layer planes and cell spheres."""
        self.fbo.use()
        self.ctx.clear(0.02, 0.02, 0.04, 1.0)
        self.update_from_state(state)

        width, height = self.color_texture.size
        aspect = width / max(height, 1)
        field_size = 1.0
        if self.config:
            field_size = self.config.retina.field_size_deg
        z_vals = layer_z_positions()
        layer_z_map = {
            "Stimulus": z_vals[0], "Cones": z_vals[1], "Horizontal": z_vals[2],
            "Bipolar": z_vals[3], "Amacrine": z_vals[4], "RGC": z_vals[5],
        }
        model = np.eye(4, dtype=np.float32)
        mvp = self.camera.mvp(model, aspect)

        for plane in self.layer_planes.values():
            plane.draw_3d(mvp)

        if self.show_cells and self.cell_spheres is None:
            self.cell_spheres = CellSpheresRenderer(
                ctx=self.ctx, layer_z=layer_z_map, subsample=4
            )
        if self.show_cells and self.cell_spheres is not None:
            self.cell_spheres.draw(state, mvp, field_size)

        data = self.color_texture.read()
        img = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))
        return np.flipud(img)

