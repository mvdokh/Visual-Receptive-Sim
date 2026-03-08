"""
RGC Circuit Simulator — Python entry point.

This wires together:
- Simulation state and pipeline
- ModernGL 3D rendering
- Dear PyGui GUI
"""

from src.gui.app import run_app


def main() -> None:
    """Start the retinal ganglion cell circuit simulator GUI."""
    run_app()


if __name__ == "__main__":
    main()

