"""
Per-layer helpers for the retinal simulation.

The current implementation keeps most of the math inside `pipeline.tick`.
These modules exist to mirror the intended project structure and are the
right place to move layer-specific computations into smaller functions
over time (e.g. `cones.compute_cone_responses`, `horizontal.apply_feedback`).
"""

