"""
RGC Circuit Simulator — Python entry point.

This wires together:
- Simulation state and pipeline
- 2D layer visualizations (single layer and all-layers mosaic)
- Dear PyGui GUI
"""

from src.gui.app import run_app


def main() -> None:
    """Start the retinal ganglion cell circuit simulator GUI."""
    run_app()


if __name__ == "__main__":
    main()

