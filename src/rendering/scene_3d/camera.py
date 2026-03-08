"""Orbit camera for 3D retinal stack / Signal Flow Column viewing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


ELEVATION_MAX = 1.483  # ~85 deg, avoid flipping

# Default: 3/4 perspective (azimuth 30°, elevation 25°) - shows top + front face
DEFAULT_AZIMUTH = np.radians(30.0)
DEFAULT_ELEVATION = np.radians(25.0)


@dataclass
class OrbitCamera:
    """Orbit camera: target, distance, azimuth, elevation. Smooth lerp toward target."""

    target: np.ndarray = field(default_factory=lambda: np.array([0.0, 4.0, 0.0], dtype=np.float32))
    distance: float = 6.0
    azimuth: float = float(DEFAULT_AZIMUTH)
    elevation: float = float(DEFAULT_ELEVATION)
    fov: float = 45.0  # degrees

    # Smoothed camera: lerp current toward target each frame
    _azimuth_target: float = field(default=0.0, init=False)
    _elevation_target: float = field(default=0.0, init=False)
    _distance_target: float = field(default=0.0, init=False)
    _lerp_rate: float = 0.12  # per-frame lerp toward target

    def __post_init__(self) -> None:
        self._azimuth_target = self.azimuth
        self._elevation_target = self.elevation
        self._distance_target = self.distance

    def set_target_from_current(self) -> None:
        """Sync target to current values."""
        self._azimuth_target = self.azimuth
        self._elevation_target = self.elevation
        self._distance_target = self.distance

    def integrate(self, dt: float) -> None:
        """Smooth lerp current toward target."""
        r = 1.0 - (1.0 - self._lerp_rate) ** (dt * 60)
        self.azimuth += (self._azimuth_target - self.azimuth) * r
        self.elevation += (self._elevation_target - self.elevation) * r
        self.distance += (self._distance_target - self.distance) * r
        self.elevation = max(-ELEVATION_MAX, min(ELEVATION_MAX, self.elevation))

    def add_drag(self, dx: float, dy: float, sensitivity: float = 0.25) -> None:
        """Add mouse drag to target angles. sensitivity multiplies raw delta."""
        self._azimuth_target -= dx * sensitivity * 0.005
        self._elevation_target += dy * sensitivity * 0.005
        self._elevation_target = max(-ELEVATION_MAX, min(ELEVATION_MAX, self._elevation_target))

    def add_zoom(self, delta: float) -> None:
        """Add scroll zoom to target distance."""
        factor = 0.96 if delta > 0 else 1.04
        self._distance_target = max(3.0, min(14.0, self._distance_target * factor))

    def set_preset(self, name: str) -> None:
        """Apply a camera preset: top, front, iso."""
        if name == "top":
            self._azimuth_target, self._elevation_target, self._distance_target = 0.0, ELEVATION_MAX, 7.0
        elif name == "front":
            self._azimuth_target, self._elevation_target, self._distance_target = 0.0, np.radians(5.0), 7.0
        elif name == "iso":
            self._azimuth_target = DEFAULT_AZIMUTH
            self._elevation_target = DEFAULT_ELEVATION
            self._distance_target = 6.0

    def eye_position(self) -> np.ndarray:
        """Camera position in world space."""
        x = self.distance * np.cos(self.elevation) * np.sin(self.azimuth)
        y = self.distance * np.sin(self.elevation)
        z = self.distance * np.cos(self.elevation) * np.cos(self.azimuth)
        return self.target + np.array([x, y, z], dtype=np.float32)

    def view_matrix(self) -> np.ndarray:
        """4x4 view matrix (column-major for OpenGL)."""
        eye = self.eye_position()
        fwd = self.target - eye
        fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(fwd, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, fwd)
        m = np.eye(4, dtype=np.float32)
        m[0:3, 0] = right
        m[0:3, 1] = up
        m[0:3, 2] = -fwd
        m[:3, 3] = -np.array([
            np.dot(right, eye), np.dot(up, eye), np.dot(-fwd, eye),
        ], dtype=np.float32)
        return m

    def projection_matrix(self, aspect: float, near: float = 0.1, far: float = 20.0) -> np.ndarray:
        """4x4 perspective projection matrix (column-major)."""
        f = 1.0 / np.tan(np.radians(self.fov) / 2.0)
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = f / aspect
        m[1, 1] = f
        m[2, 2] = (far + near) / (near - far)
        m[2, 3] = (2 * far * near) / (near - far)
        m[3, 2] = -1.0
        return m

    def mvp(self, model: np.ndarray, aspect: float) -> np.ndarray:
        """Model-View-Projection matrix."""
        v = self.view_matrix()
        p = self.projection_matrix(aspect)
        return (p @ v @ model).astype(np.float32)
