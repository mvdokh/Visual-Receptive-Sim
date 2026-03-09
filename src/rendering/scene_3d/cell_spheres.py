"""
Instanced sphere rendering for individual cells. Signal Flow Column: inside slabs,
activity-scaled size, bloom. Cell counts and rod:cone / L:M:S ratios follow
src.simulation.bio_constants (Curcio et al. 1990/1991, Masland 2012, Masland & Raviola 1998).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import moderngl
import numpy as np

from src.simulation.bio_constants import RELATIVE_DENSITY, ROD_CONE_RATIO

# Rod:cone ~20:1 — 92M rods, 4.6M cones (Curcio et al. 1990). Cone L:M:S ≈ 64:32:2 (Curcio et al. 1991).
# Overall photoreceptor→RGC convergence ~100:1 (Masland 2012). INL: ~41% bipolar, ~39% amacrine, ~3% horizontal.


class CellType(Enum):
    L_CONE = auto()
    M_CONE = auto()
    S_CONE = auto()
    HORIZONTAL = auto()
    BIPOLAR_ON = auto()
    BIPOLAR_OFF = auto()
    AMACRINE = auto()
    MIDGET_ON = auto()
    MIDGET_OFF = auto()
    PARASOL_ON = auto()
    PARASOL_OFF = auto()


# Cone units: L = warm red, M = green, S = blue (64:32:2 proportion in RELATIVE_DENSITY)
CELL_COLORS = {
    CellType.L_CONE: (1.0, 0.3, 0.1),
    CellType.M_CONE: (0.3, 1.0, 0.2),
    CellType.S_CONE: (0.2, 0.3, 1.0),
    CellType.HORIZONTAL: (0.9, 0.9, 0.3),
    CellType.BIPOLAR_ON: (1.0, 0.6, 0.0),
    CellType.BIPOLAR_OFF: (0.3, 0.6, 1.0),
    CellType.AMACRINE: (0.7, 0.3, 0.9),
    CellType.MIDGET_ON: (1.0, 0.2, 0.2),
    CellType.MIDGET_OFF: (0.2, 0.4, 1.0),
    CellType.PARASOL_ON: (1.0, 0.8, 0.0),
    CellType.PARASOL_OFF: (0.0, 0.8, 1.0),
}


def _make_icosphere(divisions: int = 1) -> tuple[np.ndarray, np.ndarray]:
    t = (1.0 + 5**0.5) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=np.float32)
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    for _ in range(divisions):
        new_faces = []
        for a, b, c in faces:
            ab = (verts[a] + verts[b]) / 2
            bc = (verts[b] + verts[c]) / 2
            ca = (verts[c] + verts[a]) / 2
            ab = ab / np.linalg.norm(ab)
            bc = bc / np.linalg.norm(bc)
            ca = ca / np.linalg.norm(ca)
            n = len(verts)
            verts = np.vstack([verts, ab, bc, ca])
            new_faces.extend([
                (a, n, n + 2), (n, b, n + 1), (n + 2, n + 1, c), (n, n + 1, n + 2)
            ])
        faces = new_faces
    for i in range(len(verts)):
        verts[i] = verts[i] / np.linalg.norm(verts[i])
    indices = np.array([f for tri in faces for f in tri], dtype=np.uint32)
    return verts.astype(np.float32), indices


# Signal Flow Column: layer name -> (y_center, thickness). Densities from bio_constants.RELATIVE_DENSITY.
SLAB_Y = {
    "Cones": (4.7, 0.6),
    "Horizontal": (4.08, 0.25),
    "Bipolar": (3.58, 0.45),
    "Amacrine": (3.38, 0.35),
    "RGC": (2.85, 0.5),
}


@dataclass
class CellSpheresRenderer:
    """Renders subsampled grid cells as instanced spheres. Inside slabs, radius by activity, optional bloom."""

    ctx: moderngl.Context
    subsample: int = 8
    radius_min: float = 0.008
    radius_max: float = 0.025

    _vao: Optional[moderngl.VertexArray] = field(default=None, init=False)
    _instance_buffer: Optional[moderngl.Buffer] = field(default=None, init=False)

    def _ensure_resources(self) -> None:
        if self._vao is not None:
            return
        verts, indices = _make_icosphere(0)
        vbo = self.ctx.buffer(verts.tobytes())
        ibo = self.ctx.buffer(indices.tobytes())
        self._instance_buffer = self.ctx.buffer(reserve=1024 * 1024)
        prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos;
                in float in_x;
                in float in_y;
                in float in_z;
                in float in_radius;
                in vec3 in_color;
                in float in_activity;
                uniform mat4 u_mvp;
                uniform float u_radius_scale;
                out vec3 v_color;
                out float v_activity;
                void main() {
                    float r = in_radius * u_radius_scale;
                    vec4 world = vec4(in_pos * r + vec3(in_x, in_y, in_z), 1.0);
                    gl_Position = u_mvp * world;
                    v_color = in_color;
                    v_activity = in_activity;
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 v_color;
                in float v_activity;
                uniform float u_alpha;
                out vec4 fragColor;
                void main() {
                    float glow = 0.3 + 0.7 * v_activity;
                    fragColor = vec4(v_color * glow, u_alpha);
                }
            """,
        )
        self._vao = self.ctx.vertex_array(
            prog,
            [
                (vbo, "3f", "in_pos"),
                (self._instance_buffer, "1f 1f 1f 1f 3f 1f /i",
                 "in_x", "in_y", "in_z", "in_radius", "in_color", "in_activity"),
            ],
            index_buffer=ibo,
        )

    def build_instances(self, state, field_size: float = 1.0) -> np.ndarray:
        """
        Returns (N, 8): x,y,z,radius,r,g,b,activity. Y = layer center + random offset within slab.
        Sphere counts per layer are proportional to RELATIVE_DENSITY (bio_constants) so the
        funnel from photoreceptors → RGC is visually apparent. Rods dominate cones ~20:1;
        cone L:M:S in 64:32:2 with warm red / green / blue.
        """
        h, w = state.grid_shape()
        step = self.subsample
        ny, nx = max(1, h // step), max(1, w // step)
        xs = (np.arange(nx) - nx / 2 + 0.5) * (field_size / max(nx, 1))
        zs = (np.arange(ny) - ny / 2 + 0.5) * (field_size / max(ny, 1))
        rng = np.random.default_rng(42)
        instances = []

        # Total density (RGC=1) so we can scale sphere counts; cap total spheres for performance
        total_density = (
            RELATIVE_DENSITY["rods"]
            + RELATIVE_DENSITY["cones_L"]
            + RELATIVE_DENSITY["cones_M"]
            + RELATIVE_DENSITY["cones_S"]
            + RELATIVE_DENSITY["horizontal"]
            + RELATIVE_DENSITY["bipolar"]
            + RELATIVE_DENSITY["amacrine"]
            + RELATIVE_DENSITY["rgc"]
        )
        max_spheres = 3500
        base = max_spheres / total_density

        def sample_positions(n_want: int):
            """Sample n_want (i,j) positions from the grid."""
            if n_want <= 0:
                return [], []
            n_avail = ny * nx
            if n_want >= n_avail:
                ii = np.repeat(np.arange(ny), nx)
                jj = np.tile(np.arange(nx), ny)
            else:
                idx = rng.choice(n_avail, size=n_want, replace=False)
                ii, jj = np.unravel_index(idx, (ny, nx))
            return ii, jj

        # Photoreceptor layer: rods (gray) + L/M/S cones in RELATIVE_DENSITY proportion
        n_rods = min(int(base * RELATIVE_DENSITY["rods"]), ny * nx)
        n_L = min(int(base * RELATIVE_DENSITY["cones_L"]), ny * nx)
        n_M = min(int(base * RELATIVE_DENSITY["cones_M"]), ny * nx)
        n_S = min(int(base * RELATIVE_DENSITY["cones_S"]), ny * nx)
        cone_L = state.cone_L if state.cone_L is not None else np.zeros((h, w))
        cone_M = state.cone_M if state.cone_M is not None else np.zeros((h, w))
        cone_S = state.cone_S if state.cone_S is not None else np.zeros((h, w))
        cone_max = max(1e-6, float(np.max(cone_L + cone_M + cone_S)))
        y_cone, thick_cone = SLAB_Y.get("Cones", (4.7, 0.6))
        for (name, grid, n, color) in [
            ("rod", (cone_L + cone_M) / 2, n_rods, (0.55, 0.52, 0.5)),
            ("L", cone_L, n_L, CELL_COLORS[CellType.L_CONE]),
            ("M", cone_M, n_M, CELL_COLORS[CellType.M_CONE]),
            ("S", cone_S, n_S, CELL_COLORS[CellType.S_CONE]),
        ]:
            if n <= 0 or grid is None:
                continue
            ii, jj = sample_positions(n)
            for idx in range(len(ii)):
                i, j = int(ii[idx]), int(jj[idx])
                pi, pj = min(i * step, h - 1), min(j * step, w - 1)
                act = max(0, min(1, float(grid[pi, pj]) / cone_max))
                rad = self.radius_min + (self.radius_max - self.radius_min) * act
                x = float(xs[j])
                z = float(zs[i])
                y_offset = (rng.random() - 0.5) * thick_cone * 0.8
                y = y_cone + y_offset
                r, g, b = color
                instances.append([x, y, z, rad, r, g, b, act])

        # INL and RGC layers: counts from RELATIVE_DENSITY (horizontal sparse, bipolar/amacrine denser)
        for layer_name, grid, cell_type, density_key in [
            ("Horizontal", state.h_activation, CellType.HORIZONTAL, "horizontal"),
            ("Bipolar", state.bp_diffuse_on, CellType.BIPOLAR_ON, "bipolar"),
            ("Amacrine", state.amacrine_aii, CellType.AMACRINE, "amacrine"),
            ("RGC", state.fr_midget_on_L, CellType.MIDGET_ON, "rgc"),
        ]:
            if grid is None:
                continue
            n_want = min(int(base * RELATIVE_DENSITY[density_key]), ny * nx)
            if n_want <= 0:
                continue
            y_center, thick = SLAB_Y.get(layer_name, (3.0, 0.3))
            r, g, b = CELL_COLORS.get(cell_type, (0.8, 0.8, 0.8))
            g_max = max(1e-6, float(np.max(grid)))
            ii, jj = sample_positions(n_want)
            for idx in range(len(ii)):
                i, j = int(ii[idx]), int(jj[idx])
                pi, pj = min(i * step, h - 1), min(j * step, w - 1)
                act = max(0, min(1, float(grid[pi, pj]) / g_max))
                rad = self.radius_min + (self.radius_max - self.radius_min) * act
                x = float(xs[j])
                z = float(zs[i])
                y_offset = (rng.random() - 0.5) * thick * 0.8
                y = y_center + y_offset
                instances.append([x, y, z, rad, r, g, b, act])

        return np.array(instances, dtype=np.float32) if instances else np.zeros((0, 8), dtype=np.float32)

    def draw(self, state, mvp: np.ndarray, field_size: float = 1.0) -> None:
        self._ensure_resources()
        inst = self.build_instances(state, field_size)
        if inst.size == 0:
            return
        self._instance_buffer.write(inst.tobytes())
        prog = self._vao.program
        prog["u_mvp"].write(mvp.T.tobytes())
        prog["u_radius_scale"].value = 1.0
        prog["u_alpha"].value = 0.9
        self.ctx.disable(moderngl.BLEND)
        self._vao.render(moderngl.TRIANGLES, instances=len(inst))

    def draw_bloom(self, state, mvp: np.ndarray, field_size: float = 1.0) -> None:
        """Second pass: 1.8× radius, 0.12 alpha, additive blend (GPU-only bloom)."""
        self._ensure_resources()
        inst = self.build_instances(state, field_size)
        if inst.size == 0:
            return
        self._instance_buffer.write(inst.tobytes())
        prog = self._vao.program
        prog["u_mvp"].write(mvp.T.tobytes())
        prog["u_radius_scale"].value = 1.8
        prog["u_alpha"].value = 0.12
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        self._vao.render(moderngl.TRIANGLES, instances=len(inst))
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.disable(moderngl.BLEND)
