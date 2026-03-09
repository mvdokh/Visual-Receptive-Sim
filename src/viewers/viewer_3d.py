"""
Vispy-based 3D viewer for RGC Circuit Simulator.

Renders to a buffer for embedding in the main app (Dear PyGui). PyMOL-style
camera (orbit/pan/zoom via add_drag/add_zoom), instanced Markers, circuit
tracing mode, scale bar. Replaces the old ModernGL 3D viewer when in 3D Stack mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import time

from src.config import GlobalConfig, Viewer3DConfig
from src.simulation.bio_constants import UM_PER_WORLD_UNIT, SCALE_BAR_UM
from src.simulation.state import SimState
from src.viewers.layer_manager import LAYER_Z, LAYER_THICKNESS, LayerManager
from src.viewers.circuit_tracer import (
    CircuitTree,
    CircuitTracer,
    ConnectionSegment,
    ConnectionType,
    CONN_AMACRINE_LATERAL,
    CONN_EXCITATORY,
    CONN_INHIBITORY,
)
from src.viewers.oscilloscope import RollingBuffer, OscilloscopeRenderer
from src.rendering.heatmap import spectrum_to_stimulus_rgba

# Cell colors: (inactive_rgb, active_rgb) 0-1
CELL_COLORS: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {
    "cone_L": ((0.784, 0.294, 0.192), (1.0, 0.6, 0.4)),
    "cone_M": ((0.227, 0.49, 0.267), (0.533, 1.0, 0.6)),
    "cone_S": ((0.169, 0.31, 0.78), (0.533, 0.733, 1.0)),
    "horizontal": ((0.545, 0.412, 0.078), (1.0, 0.851, 0.4)),
    "bipolar": ((0.102, 0.478, 0.29), (0.267, 1.0, 0.667)),
    "bipolar_off": ((0.478, 0.102, 0.29), (1.0, 0.267, 0.667)),
    "amacrine_aii": ((0.29, 0.29, 0.541), (0.667, 0.667, 1.0)),
    "amacrine_wide": ((0.416, 0.165, 0.416), (0.867, 0.533, 1.0)),
    "rgc_midget": ((0.722, 0.525, 0.043), (1.0, 0.878, 0.2)),
    "rgc_parasol": ((0.545, 0.102, 0.102), (1.0, 0.333, 0.333)),
}

# Biological display ratios for rendered cell counts (relative units).
# These are normalized per frame to hit viewer_3d.max_display_cells.
CELL_COUNT_RATIOS: Dict[str, float] = {
    # Cones split by L/M/S fractions of total cone ratio (4.6)
    "cone_L": 4.6 * 0.64,
    "cone_M": 4.6 * 0.32,
    "cone_S": 4.6 * 0.04,
    # Inner retinal layers (no rods in current simulation grid)
    "horizontal": 0.2,
    "bipolar": 2.5,
    "amacrine_aii": 2.3,
    "rgc_midget": 1.0,
}

DISPLAY_TO_POSITION_KEY = {
    "cone_L": "cone_L",
    "cone_M": "cone_M",
    "cone_S": "cone_S",
    "horizontal": "horizontal",
    "bipolar": "bipolar_on_diffuse",
    "amacrine_aii": "amacrine_aii",
    "rgc_midget": "rgc_midget_on",
}


def _lerp_color(
    inactive: Tuple[float, float, float],
    active: Tuple[float, float, float],
    rate: float,
) -> Tuple[float, float, float, float]:
    r = inactive[0] + (active[0] - inactive[0]) * rate
    g = inactive[1] + (active[1] - inactive[1]) * rate
    b = inactive[2] + (active[2] - inactive[2]) * rate
    a = 0.6 + 0.4 * min(1.0, rate + 0.01)
    return (r, g, b, a)


def _build_scale_bar_geometry(
    length_wu: float,
    center_xyz: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    cx, cy, cz = center_xyz
    half = length_wu / 2
    verts = np.array([
        [cx, cy, cz - half],
        [cx, cy, cz + half],
    ], dtype=np.float32)
    colors = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ], dtype=np.float32)
    return verts, colors


def _even_grid_indices(
    h: int,
    w: int,
    n: int,
) -> np.ndarray:
    """
    Evenly sample n flat indices across an (h, w) grid using a regular lattice.

    This preserves an even distribution while avoiding per-frame randomness.
    """
    n = int(max(0, n))
    if n == 0 or h <= 0 or w <= 0:
        return np.zeros((0,), dtype=np.int64)

    total = h * w
    if n >= total:
        return np.arange(total, dtype=np.int64)

    aspect_ratio = float(w) / float(h) if h > 0 else 1.0
    cols = int(np.ceil(np.sqrt(n * aspect_ratio)))
    cols = max(1, cols)
    rows = int(np.ceil(float(n) / float(cols)))
    rows = max(1, rows)

    xs = np.linspace(0.0, float(w - 1), cols, dtype=np.float32)
    ys = np.linspace(0.0, float(h - 1), rows, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)
    coords = np.stack([YY, XX], axis=-1).reshape(-1, 2)[:n]

    ij = np.rint(coords).astype(np.int64)
    ij[:, 0] = np.clip(ij[:, 0], 0, h - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, w - 1)
    flat = ij[:, 0] * w + ij[:, 1]
    return flat.astype(np.int64, copy=False)


try:
    from vispy import app as vispy_app
    from vispy.scene import SceneCanvas
    from vispy.scene.visuals import Markers, Line, Mesh, Text
    from vispy.scene.cameras import TurntableCamera
    from vispy.visuals.transforms import STTransform
    HAS_VISPY = True
except ImportError:
    HAS_VISPY = False


@dataclass
class Viewer3DState:
    layer_manager: Optional[LayerManager] = None
    circuit_tracer: Optional[CircuitTracer] = None
    circuit_tree: Optional[CircuitTree] = None
    spherical_mode: bool = False
    max_density: bool = False
    density_warning: bool = False
    # Cached layout / LOD state
    grid_h: int = 0
    grid_w: int = 0
    max_display_cells: int = 0
    display_indices: Dict[str, np.ndarray] = field(default_factory=dict)
    display_jitter: Dict[str, np.ndarray] = field(default_factory=dict)


class VispyViewer3D:
    """
    Vispy 3D viewer that renders to a buffer. Used by the main app when
    view mode is "3D Stack". No Qt; camera controlled via add_drag / add_zoom.
    """

    def __init__(
        self,
        size: Tuple[int, int],
        config: Optional[GlobalConfig] = None,
    ):
        if not HAS_VISPY:
            raise RuntimeError("vispy is required for 3D viewer. pip install vispy")
        self._size = size
        self._config = config
        self._viewer_cfg = (config.viewer_3d if config else Viewer3DConfig())
        self._state = Viewer3DState()

        # Use glfw backend so we don't require Qt
        try:
            vispy_app.use_app("glfw")
        except Exception:
            pass
        self._canvas = SceneCanvas(size=size, keys="interactive")
        self._scene = self._canvas.central_widget.add_view()
        self._scene.bgcolor = (0.02, 0.02, 0.06, 1.0)

        self._camera = TurntableCamera(
            fov=self._viewer_cfg.camera_fov,
            distance=12.0,
            elevation=20.0,
            azimuth=45.0,
        )
        self._scene.camera = self._camera

        self._markers: Dict[str, Markers] = {}
        self._line_visual: Optional[Line] = None
        self._scale_bar_line: Optional[Line] = None

        # Layer slabs (translucent quads per anatomical layer, flat mode only).
        self._slab_meshes: Dict[str, Mesh] = {}

        # LOD state (sprite size transitions)
        self._lod_tier: str = "close"
        self._lod_prev_tier: str = "close"
        self._lod_transition_start: float = time.perf_counter()
        self._lod_prev_size_mult: float = 1.0
        self._lod_size_mult: float = 1.0

        # Last full-resolution positions per layer (for circuit tracing / picking).
        self._positions_3d: Dict[str, np.ndarray] = {}

        # Oscilloscope state per coarse layer.
        self._osc_renderer = OscilloscopeRenderer(width_px=120, height_px=40)
        self._osc_mean: Dict[str, RollingBuffer] = {}
        self._osc_sel: Dict[str, RollingBuffer] = {}
        self._osc_images: Dict[str, Any] = {}

        # Stimulus display slab (image + label).
        self._stim_image: Any = None
        self._stim_label: Any = None

    def resize(self, w: int, h: int) -> None:
        self._size = (w, h)
        self._canvas.size = (w, h)

    def add_drag(self, dx: float, dy: float, sensitivity: float = 0.18) -> None:
        """Orbit camera (called from main app when left-dragging viewport)."""
        self._camera.orbit(-dx * sensitivity, dy * sensitivity)

    def add_zoom(self, delta: float) -> None:
        """Zoom camera (called when scroll over viewport)."""
        self._camera.distance = max(1.0, self._camera.distance * (1.0 - float(delta) * 0.1))

    def set_selection_from_grid(
        self,
        coarse_layer: str,
        grid_x: float,
        grid_y: float,
    ) -> None:
        """
        Set the current circuit selection based on a coarse layer name and grid coords.

        coarse_layer comes from the 2D/3D pick combo (\"RGC\", \"Cone\", etc.).
        """
        if self._state.circuit_tracer is None or not self._positions_3d:
            return
        lm = self._state.layer_manager
        if lm is None:
            return
        h, w = lm.grid_h, lm.grid_w
        if h <= 0 or w <= 0:
            return

        # Map coarse pick layer → fine-grained layer key used in CircuitTracer.
        coarse = coarse_layer.strip().lower()
        if coarse.startswith("rgc"):
            picked_layer = "rgc_midget_on"
        elif coarse.startswith("cone"):
            picked_layer = "cone_L"
        elif coarse.startswith("bipolar"):
            picked_layer = "bipolar_on_diffuse"
        elif coarse.startswith("horizontal"):
            picked_layer = "horizontal"
        elif coarse.startswith("amacrine"):
            picked_layer = "amacrine_aii"
        else:
            picked_layer = "rgc_midget_on"

        i = int(np.clip(int(round(grid_y)), 0, h - 1))
        j = int(np.clip(int(round(grid_x)), 0, w - 1))

        try:
            tree = self._state.circuit_tracer.resolve(
                picked_layer,
                i,
                j,
                self._positions_3d,
            )
        except Exception:
            return
        self._state.circuit_tree = tree

    def clear_selection(self) -> None:
        """Clear any active circuit selection."""
        self._state.circuit_tree = None

    def _update_lod(self) -> Tuple[float, str]:
        """
        Update LOD tier based on camera distance and return (size_mult, tier).

        LOD does not change opacity; only sprite size and rendered counts.
        """
        dist = float(self._camera.distance)
        if dist < 3.0:
            tier = "close"
            target_mult = 1.0
        elif dist <= 10.0:
            tier = "mid"
            target_mult = 0.7
        else:
            tier = "far"
            target_mult = 0.4

        now = time.perf_counter()
        # Initialize state on first call
        if self._lod_tier is None:
            self._lod_tier = tier
            self._lod_prev_tier = tier
            self._lod_prev_size_mult = target_mult
            self._lod_size_mult = target_mult
            self._lod_transition_start = now
            return self._lod_size_mult, tier

        if tier != self._lod_tier:
            self._lod_prev_tier = self._lod_tier
            self._lod_prev_size_mult = self._lod_size_mult
            self._lod_tier = tier
            self._lod_transition_start = now

        # Smooth transition over 0.2 seconds
        dt = max(0.0, now - self._lod_transition_start)
        alpha = min(1.0, dt / 0.2) if self._lod_prev_tier != self._lod_tier else 1.0
        self._lod_size_mult = (
            self._lod_prev_size_mult
            + (target_mult - self._lod_prev_size_mult) * float(alpha)
        )
        return self._lod_size_mult, tier

    def _ensure_display_layout(
        self,
        h: int,
        w: int,
        active_layers: List[str],
    ) -> None:
        """
        Ensure per-layer display indices/jitter are computed for the current grid.

        Uses viewer_3d.max_display_cells and CELL_COUNT_RATIOS to allocate counts.
        """
        max_cells = int(getattr(self._viewer_cfg, "max_display_cells", 8000))
        # Rebuild when grid size or target count changes, or if layout empty.
        if (
            self._state.display_indices
            and self._state.grid_h == h
            and self._state.grid_w == w
            and self._state.max_display_cells == max_cells
        ):
            return

        self._state.grid_h = h
        self._state.grid_w = w
        self._state.max_display_cells = max_cells
        self._state.display_indices.clear()
        self._state.display_jitter.clear()

        # Only consider layers we actually have data for and a ratio for.
        ratios: Dict[str, float] = {}
        for key in active_layers:
            if key in CELL_COUNT_RATIOS:
                ratios[key] = float(CELL_COUNT_RATIOS[key])
        if not ratios:
            return

        total_ratio = float(sum(ratios.values()))
        if total_ratio <= 0.0:
            return

        rng = np.random.default_rng(12345)
        used = 0
        keys = list(ratios.keys())
        n_cells_total = h * w

        for i, key in enumerate(keys):
            if i == len(keys) - 1:
                # Last layer gets any remaining budget.
                remaining = max_cells - used
                n_target = max(0, remaining)
            else:
                frac = ratios[key] / total_ratio
                n_target = int(max_cells * frac)

            n_target = max(0, min(n_target, n_cells_total))
            if n_target == 0:
                continue

            idx = _even_grid_indices(h, w, n_target)
            if idx.size == 0:
                continue

            jitter = (rng.random((idx.shape[0], 2), dtype=np.float32) * 0.04) - 0.02
            self._state.display_indices[key] = idx.astype(np.int64, copy=False)
            self._state.display_jitter[key] = jitter.astype(np.float32, copy=False)
            used += idx.shape[0]

    def _update_slabs(
        self,
        half_width: float,
        half_height: float,
        spherical: bool,
    ) -> None:
        """
        Draw faint translucent rectangular slabs for each coarse anatomical layer.

        Slabs are visual scaffolding only and are only shown in flat mode.
        """
        if spherical:
            # Hide slabs in spherical mode to avoid mismatched flat planes.
            for mesh in self._slab_meshes.values():
                mesh.visible = False
            return

        # Coarse layer centers and representative color keys.
        coarse_layers = [
            ("rgc", "rgc_midget"),
            ("amacrine", "amacrine_aii"),
            ("bipolar", "bipolar"),
            ("horizontal", "horizontal"),
            ("cone", "cone_L"),
        ]

        for layer_name, color_key in coarse_layers:
            thickness = float(LAYER_THICKNESS.get(layer_name, 0.0))
            # Use corresponding fine-grained Z centers from LAYER_Z.
            if layer_name == "rgc":
                z_center = float(LAYER_Z.get("rgc_midget_on", 0.0))
            elif layer_name == "amacrine":
                z_center = float(LAYER_Z.get("amacrine_aii", 0.0))
            elif layer_name == "bipolar":
                z_center = float(LAYER_Z.get("bipolar_on_diffuse", 0.0))
            elif layer_name == "horizontal":
                z_center = float(LAYER_Z.get("horizontal", 0.0))
            elif layer_name == "cone":
                z_center = float(LAYER_Z.get("cone_L", 0.0))
            else:
                z_center = 0.0

            # Simple single-quad slab at the anatomical Z center.
            x0 = -half_width
            x1 = half_width
            y0 = -half_height
            y1 = half_height
            z = z_center
            verts = np.array(
                [
                    [x0, y0, z],
                    [x1, y0, z],
                    [x1, y1, z],
                    [x0, y1, z],
                ],
                dtype=np.float32,
            )
            faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)

            inactive, active = CELL_COLORS.get(
                color_key, ((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
            )
            base = np.array(inactive, dtype=np.float32) * 0.5 + np.array(
                active, dtype=np.float32
            ) * 0.5
            color = np.concatenate(
                [base * 0.2, np.array([0.06], dtype=np.float32)], axis=0
            )

            mesh = self._slab_meshes.get(layer_name)
            if mesh is None:
                mesh = Mesh(
                    vertices=verts,
                    faces=faces,
                    color=tuple(color.tolist()),
                    parent=self._scene.scene,
                )
                self._slab_meshes[layer_name] = mesh
            else:
                mesh.set_data(vertices=verts, faces=faces, color=tuple(color.tolist()))
            mesh.visible = True

    def _ensure_osc_buffers(self) -> None:
        """Lazily allocate oscilloscope buffers for each coarse layer."""
        for name in ("RGC", "Amacrine", "Bipolar", "Horizontal", "Cone"):
            if name not in self._osc_mean:
                self._osc_mean[name] = RollingBuffer(capacity=200)
            if name not in self._osc_sel:
                self._osc_sel[name] = RollingBuffer(capacity=200)

    def _update_oscilloscopes(
        self,
        state: SimState,
        spherical: bool,
        half_width: float,
    ) -> None:
        """
        Update per-layer oscilloscope panels: population mean and selected-cell traces.
        """
        if not HAS_VISPY:
            return
        self._ensure_osc_buffers()

        # Map coarse osc layers to simulation arrays and colors.
        layer_map = {
            "Cone": (state.cone_L, CELL_COLORS["cone_L"][1]),
            "Horizontal": (state.h_activation, CELL_COLORS["horizontal"][1]),
            "Bipolar": (state.bp_diffuse_on, CELL_COLORS["bipolar"][1]),
            "Amacrine": (state.amacrine_aii, CELL_COLORS["amacrine_aii"][1]),
            "RGC": (state.fr_midget_on_L, CELL_COLORS["rgc_midget"][1]),
        }

        # Current selected cell (if any)
        picked_layer = None
        picked_index = None
        if self._state.circuit_tree is not None:
            picked_layer = self._state.circuit_tree.picked_layer
            picked_index = self._state.circuit_tree.picked_index

        lm = self._state.layer_manager
        grid_h = lm.grid_h if lm is not None else state.grid_shape()[0]
        grid_w = lm.grid_w if lm is not None else state.grid_shape()[1]

        for osc_name, (grid, active_rgb) in layer_map.items():
            mean_buf = self._osc_mean[osc_name]
            sel_buf = self._osc_sel[osc_name]

            if grid is None:
                mean_buf.append(0.0)
                sel_buf.clear()
                continue

            mean_val = float(np.mean(grid.astype(np.float32)))
            mean_buf.append(mean_val)

            # Selected-cell trace only for matching anatomical layer.
            sel_layer_match = False
            if picked_layer is not None and picked_index is not None:
                if osc_name == "RGC" and picked_layer.startswith("rgc"):
                    sel_layer_match = True
                elif osc_name == "Cone" and picked_layer.startswith("cone"):
                    sel_layer_match = True
                elif osc_name == "Bipolar" and picked_layer.startswith("bipolar"):
                    sel_layer_match = True
                elif osc_name == "Horizontal" and picked_layer == "horizontal":
                    sel_layer_match = True
                elif osc_name == "Amacrine" and picked_layer.startswith("amacrine"):
                    sel_layer_match = True

            if sel_layer_match and grid_h > 0 and grid_w > 0:
                i = int(picked_index // grid_w)
                j = int(picked_index % grid_w)
                i = int(np.clip(i, 0, grid_h - 1))
                j = int(np.clip(j, 0, grid_w - 1))
                sel_val = float(grid[i, j])
                sel_buf.append(sel_val)
            else:
                sel_buf.clear()

            # Create/update panel image.
            rgba = self._osc_renderer.render(
                mean_buf,
                sel_buf if sel_buf.values().size > 0 else None,
                mean_color=(*active_rgb, 1.0),
                overlay_color=(1.0, 1.0, 1.0, 1.0),
            )

            img_visual = self._osc_images.get(osc_name)
            if img_visual is None:
                # Place panels to the right of the retinal field, one per layer at its Z.
                if osc_name == "RGC":
                    z = float(LAYER_Z.get("rgc_midget_on", 0.0))
                elif osc_name == "Amacrine":
                    z = float(LAYER_Z.get("amacrine_aii", 0.0))
                elif osc_name == "Bipolar":
                    z = float(LAYER_Z.get("bipolar_on_diffuse", 0.0))
                elif osc_name == "Horizontal":
                    z = float(LAYER_Z.get("horizontal", 0.0))
                elif osc_name == "Cone":
                    z = float(LAYER_Z.get("cone_L", 0.0))
                else:
                    z = 0.0
                # Panel world size: 1.5 (width) x 0.5 (height)
                panel_w = 1.5
                panel_h = 0.5
                x_center = half_width + 0.3
                y_center = 0.0
                x = x_center - panel_w * 0.5
                y = y_center - panel_h * 0.5

                from vispy.scene.visuals import Image  # local import to avoid optional dep issues

                img_visual = Image(
                    rgba,
                    parent=self._scene.scene,
                    interpolation="nearest",
                )
                sx = panel_w / float(self._osc_renderer.width)
                sy = panel_h / float(self._osc_renderer.height)
                img_visual.transform = STTransform(
                    scale=(sx, sy, 1.0),
                    translate=(x, y, z),
                )
                self._osc_images[osc_name] = img_visual
            else:
                img_visual.set_data(rgba)
    def update_frame(self, state: SimState) -> None:
        """Update scene from simulation state (markers, lines, scale bar)."""
        h, w = state.grid_shape()
        dx_deg = state.config.retina.field_size_deg / w
        microns_per_px = state.config.retina.microns_per_px

        if (
            self._state.layer_manager is None
            or self._state.layer_manager.grid_h != h
            or self._state.layer_manager.grid_w != w
            or abs(float(self._state.layer_manager.dx_deg) - float(dx_deg)) > 1e-12
            or abs(float(self._state.layer_manager.microns_per_px) - float(microns_per_px)) > 1e-9
        ):
            self._state.layer_manager = LayerManager(
                grid_h=h, grid_w=w,
                dx_deg=dx_deg,
                microns_per_px=microns_per_px,
                layer_spacing=self._viewer_cfg.layer_spacing,
                spherical_radius_mm=self._viewer_cfg.spherical_radius_mm,
            )
            self._state.circuit_tracer = CircuitTracer(h, w, microns_per_px)

        lm = self._state.layer_manager
        spherical = self._state.spherical_mode
        size_mult, lod_tier = self._update_lod()

        layer_data = [
            ("cone_L", state.cone_L),
            ("cone_M", state.cone_M),
            ("cone_S", state.cone_S),
            ("horizontal", state.h_activation),
            ("bipolar", state.bp_diffuse_on),
            ("amacrine_aii", state.amacrine_aii),
            ("rgc_midget", state.fr_midget_on_L),
        ]

        # Active display layers for this frame (with data and geometry).
        active_layers: List[str] = []
        for layer_key, grid in layer_data:
            if grid is None:
                continue
            pos_key = DISPLAY_TO_POSITION_KEY.get(layer_key, layer_key)
            if pos_key not in LAYER_Z:
                continue
            active_layers.append(layer_key)

        self._ensure_display_layout(h, w, active_layers)

        positions_3d: Dict[str, np.ndarray] = {}
        # Compute XY extents from the first available layer for slab sizing.
        half_width = 1.0
        half_height = 1.0
        for layer_key, grid in layer_data:
            if grid is None:
                continue
            pos_key = DISPLAY_TO_POSITION_KEY.get(layer_key, layer_key)
            if pos_key not in LAYER_Z:
                continue

            # Use full-resolution positions for connectivity/picking; subsample with
            # explicit index sets for display so we can hit max_display_cells and
            # preserve an even spatial distribution.
            pos_full = positions_3d.get(pos_key)
            if pos_full is None:
                pos_full = lm.get_positions(pos_key, spherical=spherical, subsample=1)
                positions_3d[pos_key] = pos_full
                # Use the first populated layer to infer XY extent (centered grid).
                if pos_full.size:
                    half_width = float(np.max(np.abs(pos_full[:, 0])))
                    half_height = float(np.max(np.abs(pos_full[:, 1])))

            display_idx = self._state.display_indices.get(layer_key)
            jitter = self._state.display_jitter.get(layer_key)
            if display_idx is None or jitter is None or display_idx.size == 0:
                continue

            # LOD: at far tier, half the rendered count unless max_density override is on.
            if lod_tier == "far" and not self._state.max_density and display_idx.size > 1:
                idx = display_idx[::2]
                jitter_layer = jitter[::2]
            else:
                idx = display_idx
                jitter_layer = jitter

            if idx.size == 0:
                continue

            pos = pos_full[idx].astype(np.float32, copy=True)
            # Small per-cell jitter in XY so lattice doesn't look too rigid.
            pos[:, 0] += jitter_layer[:, 0]
            pos[:, 1] += jitter_layer[:, 1]

            grid_flat_full = grid.ravel()
            if grid_flat_full.size != h * w:
                grid_flat_full = np.resize(grid_flat_full, h * w)
            grid_flat = grid_flat_full[idx]

            rate = np.clip(grid_flat.astype(np.float32), 0.0, 1.0)
            # If everything is exactly zero, still render faint cells so the user
            # sees the layer geometry when 3D first opens.
            if not np.any(rate):
                rate[...] = 0.3
            inactive, active = CELL_COLORS.get(layer_key, ((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)))
            inactive3 = np.array(inactive, dtype=np.float32)[None, :]
            active3 = np.array(active, dtype=np.float32)[None, :]
            rgb = inactive3 + (active3 - inactive3) * rate[:, None]
            alpha = (0.6 + 0.4 * np.minimum(1.0, rate + 0.01)).astype(np.float32)
            colors_rgba = np.concatenate([rgb, alpha[:, None]], axis=1).astype(np.float32, copy=False)

            if self._state.circuit_tree is not None:
                cells_in = self._state.circuit_tree.cells_in_circuit
                pk = DISPLAY_TO_POSITION_KEY.get(layer_key, layer_key)
                for i, flat_idx in enumerate(idx.tolist()):
                    if (layer_key, flat_idx) not in cells_in and (pk, flat_idx) not in cells_in:
                        colors_rgba[i, :3] *= 0.3
                        colors_rgba[i, 3] *= 0.15

            if layer_key not in self._markers:
                self._markers[layer_key] = Markers(
                    parent=self._scene.scene,
                    size=5.0,
                    scaling=True,
                    alpha=0.9,
                )
            base_size = 5.0
            self._markers[layer_key].set_data(
                pos,
                edge_color=None,
                face_color=colors_rgba,
                size=base_size * float(size_mult),
            )

        # Cache full-resolution positions for circuit tracing / picking.
        self._positions_3d = positions_3d

        # Stimulus slab: use same spectral colormap as 2D viewer.
        if state.stimulus_spectrum is not None:
            try:
                wl = state.config.spectral.wavelengths
                rgba = spectrum_to_stimulus_rgba(
                    state.stimulus_spectrum.astype(np.float32),
                    wl,
                ).astype(np.float32)
                # Apply overall opacity for 3D slab.
                if rgba.shape[2] == 4:
                    rgba[..., 3] = 0.9

                from vispy.scene.visuals import Image  # local import

                if self._stim_image is None:
                    z = float(LAYER_Z.get("stimulus", 5.5))
                    stim_h, stim_w = rgba.shape[0], rgba.shape[1]
                    panel_w = 2.0 * half_width
                    panel_h = 2.0 * half_height
                    sx = panel_w / float(max(stim_w, 1))
                    sy = panel_h / float(max(stim_h, 1))
                    x = -half_width
                    y = -half_height
                    self._stim_image = Image(rgba, parent=self._scene.scene, interpolation="bilinear")
                    self._stim_image.transform = STTransform(
                        scale=(sx, sy, 1.0),
                        translate=(x, y, z),
                    )
                else:
                    self._stim_image.set_data(rgba)
            except Exception:
                pass

        # Update translucent anatomical slabs (flat mode only).
        self._update_slabs(half_width, half_height, spherical)

        # Update oscilloscope panels slightly above each slab plane so they are not coplanar.
        self._update_oscilloscopes(state, spherical, half_width)

        if self._state.circuit_tree is not None and self._state.circuit_tree.segments:
            segs = self._state.circuit_tree.segments
            verts = []
            colors = []
            for s in segs:
                verts.append(s.start_xyz)
                verts.append(s.end_xyz)
                c = CONN_EXCITATORY if s.conn_type == ConnectionType.EXCITATORY else (
                    CONN_INHIBITORY if s.conn_type == ConnectionType.INHIBITORY else CONN_AMACRINE_LATERAL
                )
                colors.append(c)
                colors.append(c)
            v = np.array(verts, dtype=np.float32)
            c = np.array(colors, dtype=np.float32)
            if self._line_visual is None:
                self._line_visual = Line(parent=self._scene.scene, width=1.5)
            self._line_visual.set_data(pos=v, color=c)
            self._line_visual.visible = True
        elif self._line_visual is not None:
            self._line_visual.visible = False

        # Z-axis scale bar: use configured scale_bar_um but show as vertical depth bar.
        scale_um = float(getattr(self._viewer_cfg, "scale_bar_um", SCALE_BAR_UM))
        scale_wu = scale_um / float(UM_PER_WORLD_UNIT)
        # Place just to the left of the retinal field, centered around mid-depth of stack.
        z_center = float(LAYER_Z.get("rgc_midget_on", 0.0)) + scale_wu * 0.5
        v, c = _build_scale_bar_geometry(scale_wu, (-half_width - 0.3, 0.0, z_center))
        if self._scale_bar_line is None:
            self._scale_bar_line = Line(parent=self._scene.scene, width=2.0)
        self._scale_bar_line.set_data(pos=v, color=c)
        self._scale_bar_line.visible = True

    def _subsample_index(self, i: int, subsample: int, h: int, w: int) -> int:
        rows = np.arange(0, h, subsample)
        cols = np.arange(0, w, subsample)
        n_per_row = len(cols)
        row = i // n_per_row
        col = i % n_per_row
        return int(rows[row] * w + cols[col])

    def render(self) -> np.ndarray:
        """
        Render the scene and return (H, W, 4) uint8 RGBA for the main app texture.
        """
        try:
            # SceneCanvas.render() returns (H, W, 4) in some vispy versions
            img = self._canvas.render(alpha=True)
        except TypeError:
            img = self._canvas.render()
        if img is None:
            # Fallback: black frame
            w, h = self._size
            return np.full((h, w, 4), 0, dtype=np.uint8)
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if img.shape[2] == 3:
            out = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
            out[:, :, :3] = img
            out[:, :, 3] = 255
            img = out
        # One-time debug dump so we can inspect what Vispy is actually rendering,
        # independent of Dear PyGui.
        if not getattr(self, "_debug_saved", False):
            try:
                Image.fromarray(img).save("debug_3d_frame.png")
            except Exception:
                pass
            self._debug_saved = True
        # Flip Y for display
        return np.ascontiguousarray(np.flipud(img))
