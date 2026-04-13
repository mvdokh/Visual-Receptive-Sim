"""3D rendering of the retinal stack via ModernGL (OpenGL context loads lazily)."""

from __future__ import annotations

__all__ = ["RenderContext"]


def __getattr__(name: str):
    if name == "RenderContext":
        from .context import RenderContext

        return RenderContext
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
