Quick start
===========

This section describes how to **run the application**. For the underlying model and equations, see the :doc:`/concepts/overview` and :doc:`/concepts/equations` sections.

Installation
------------

**Requirements**: Python 3.10+ (recommended), NumPy, SciPy, Numba, ModernGL, Dear PyGui, colour-science, Pillow, scikit-image, matplotlib, pyglm.

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate    # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   python main.py

The main window opens with a 2D heatmap view by default. Use the **View → Mode** combo to switch to **3D Stack** for the Signal Flow Column view.

Running the simulator
---------------------

- **Stimulus** is updated every frame from the current stimulus type and parameters (spot, grating, etc.). Time advances with the main loop (variable :math:`\Delta t`).
- **Pause** is not implemented; the simulation runs continuously while the window is open.
- **Export**: Use the **Export** tab in the right panel to save a screenshot (PNG), layer stats (CSV), or layer grids (NPY).

For a description of the interface (panels, 2D/3D views, connectivity, etc.), see :doc:`interface`.
