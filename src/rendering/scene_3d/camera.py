"""Orbit camera for 3D retinal stack viewing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


ELEVATION_MAX = 1.483  # ~85 deg, avoid flipping

@dataclass
class OrbitCamera:
    """Orbit camera: target, distance, azimuth, elevation. Supports damped motion."""

    target: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 3.0], dtype=np.float32))
    distance: float = 4.0
    azimuth: float = 0.4  # radians
    elevation: float = 0.5  # radians
    fov: float = 45.0  # degrees
    # Damped velocity for smooth camera inertia
    _azimuth_vel: float = 0.0
    _elevation_vel: float = 0.0
    _distance_vel: float = 0.0
    _damping: float = 0.85  # per-frame decay

    def apply_damping(self, dt: float) -> None:
        """Apply damping to camera velocity."""
        decay = self._damping ** (dt * 60)
        self._azimuth_vel *= decay
        self._elevation_vel *= decay
        self._distance_vel *= decay

    def integrate(self, dt: float) -> None:
        """Integrate velocity into azimuth, elevation, distance."""
        self.azimuth += self._azimuth_vel * dt
        self.elevation = max(-ELEVATION_MAX, min(ELEVATION_MAX, self.elevation + self._elevation_vel * dt))
        self.distance = max(2.0, min(12.0, self.distance + self._distance_vel * dt))
        self.apply_damping(dt)

    def set_preset(self, name: str) -> None:
        """Apply a camera preset: top, side, iso."""
        self._azimuth_vel = 0.0
        self._elevation_vel = 0.0
        self._distance_vel = 0.0
        if name == "top":
            self.azimuth, self.elevation, self.distance = 0.0, ELEVATION_MAX, 5.0
        elif name == "side":
            self.azimuth, self.elevation, self.distance = 0.0, 0.0, 5.0
        elif name == "iso":
            self.azimuth, self.elevation, self.distance = 0.785, 0.5, 5.0

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
        # Column-major
        m = np.eye(4, dtype=np.float32)
        m[0:3, 0] = right
        m[0:3, 1] = up
        m[0:3, 2] = -fwd
        m[0:3, 3] = eye
        # Actually standard lookAt: R = [right, up, -fwd], t = -R @ eye
        m[:3, 3] = -np.array([
            np.dot(right, eye),
            np.dot(up, eye),
            np.dot(-fwd, eye),
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
