"""
ModernGL rendering context for the 3D Signal Flow Column.

Scene: thick horizontal slabs (anatomical cross-section), connectivity lines,
cell spheres with bloom, slice plane with 1D activity graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import moderngl
import numpy as np
from scipy.ndimage import gaussian_filter

from src.config import GlobalConfig, signal_flow_slab_layout
from src.rendering.heatmap import spectrum_to_stimulus_rgba
from src.rendering.scene_3d.camera import OrbitCamera
from src.rendering.scene_3d.cell_spheres import CellSpheresRenderer
from src.rendering.scene_3d.connectivity_lines import ConnectivityLines
from src.rendering.scene_3d.signal_flow_slabs import LayerSlab, create_slabs
from src.rendering.scene_3d.slice_plane import SlicePlaneRenderer
from src.simulation.state import SimState


@dataclass
class RenderContext:
    """Signal Flow Column: slabs, connectivity, cells, slice, MSAA, bloom."""

    size: Tuple[int, int] = (1024, 1024)
    config: GlobalConfig | None = None

    ctx: moderngl.Context = field(init=False)
    fbo_msaa: Optional[moderngl.Framebuffer] = field(default=None, init=False)
    fbo_resolve: Optional[moderngl.Framebuffer] = field(default=None, init=False)

    slabs: Dict[str, LayerSlab] = field(default_factory=dict, init=False)
    camera: OrbitCamera = field(default_factory=OrbitCamera, init=False)
    cell_spheres: Optional[CellSpheresRenderer] = field(default=None, init=False)
    connectivity: Optional[ConnectivityLines] = field(default=None, init=False)
    slice_renderer: Optional[SlicePlaneRenderer] = field(default=None, init=False)

    show_cells: bool = True
    show_connectivity: bool = True
    slice_x: float = 0.0
    _slab_layout: list = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.ctx = moderngl.create_standalone_context()
        self._resize(*self.size)

    def _resize(self, width: int, height: int) -> None:
        # Single-sample FBO only (MSAA + standalone context + readback is unreliable)
        color_tex = self.ctx.texture((width, height), 4)
        depth_rb = self.ctx.depth_renderbuffer((width, height))
        self.fbo_msaa = self.ctx.framebuffer(
            color_attachments=[color_tex],
            depth_attachment=depth_rb,
        )
        self.fbo_resolve = self.fbo_msaa  # same FBO, no resolve needed
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

    @property
    def color_texture(self):
        return self.fbo_resolve.color_attachments[0]

    def ensure_scene(self, state: SimState) -> None:
        if self.slabs:
            return
        self._slab_layout = list(signal_flow_slab_layout())
        self.slabs = create_slabs(self.ctx, state)
        self.connectivity = ConnectivityLines(self.ctx, self._slab_layout, subsample=8)
        self.slice_renderer = SlicePlaneRenderer(self.ctx)
        self.cell_spheres = CellSpheresRenderer(self.ctx, subsample=4)

    def update_from_state(self, state: SimState) -> None:
        self.ensure_scene(state)
        cfg = state.config
        wl = cfg.spectral.wavelengths if cfg else np.arange(380, 701, 5, dtype=np.float32)
        stim_rgba = (
            spectrum_to_stimulus_rgba(state.stimulus_spectrum, wl)
            if state.stimulus_spectrum is not None
            else np.zeros((*state.grid_shape(), 4), dtype=np.float32)
        )
        mapping = {
            "Stimulus": stim_rgba,
            "Cones": state.cone_L,
            "Horizontal": state.h_activation,
            "Bipolar": state.bp_diffuse_on,
            "Amacrine": state.amacrine_aii,
            "RGC": state.fr_midget_on_L,
        }
        for label, slab in self.slabs.items():
            grid = mapping.get(label)
            if grid is not None:
                slab.update_from_grid(grid)

    def _bloom_pass(self, img: np.ndarray, threshold: float = 0.8) -> np.ndarray:
        """Simple bloom: 3x3 Gaussian blur on bright pixels."""
        f = img.astype(np.float32) / 255.0
        mask = np.any(f[..., :3] > threshold, axis=-1, keepdims=True)
        blurred = np.zeros_like(f)
        for c in range(3):
            ch = f[..., c]
            blurred[..., c] = gaussian_filter(ch, sigma=1.0)
        out = np.where(mask, blurred, f)
        return (np.clip(out, 0, 1) * 255).astype(np.uint8)

    def render_3d(self, state: SimState) -> np.ndarray:
        """Render Signal Flow Column, resolve MSAA, apply bloom, return uint8 RGBA."""
        self.fbo_msaa.use()
        self.ctx.clear(0.05, 0.05, 0.12, 1.0)
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

        # 2. Connectivity lines
        if self.show_connectivity and self.connectivity is not None:
            self.connectivity.draw(state, mvp)

        # 3. Cell spheres (solid)
        if self.show_cells and self.cell_spheres is not None:
            self.cell_spheres.draw(state, mvp, field_size)

        # 4. Cell spheres bloom pass
        if self.show_cells and self.cell_spheres is not None:
            self.cell_spheres.draw_bloom(state, mvp, field_size)

        # 5. Slice plane line graph
        if self.slice_renderer is not None:
            self.slice_renderer.draw(state, mvp, self.slice_x)

        self.fbo_msaa.use()
        data = self.fbo_msaa.color_attachments[0].read()
        img = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))
        img = np.flipud(img)

        # Bloom post-pass (numpy)
        img = self._bloom_pass(img, threshold=0.8)

        return img
