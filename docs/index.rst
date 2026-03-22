RGC Circuit Simulator
=====================

This project is a **retinal ganglion cell (RGC) circuit simulator**: light enters as a spectral field on a patch of retina, and the code walks it through cones, horizontal feedback, bipolars, amacrines, and finally RGC generators and firing rates. The documentation is organized around that pipeline so you can read the math next to the stage it describes, and use the UI when you want the same quantities as sliders and layer views.

Implementation: vectorized NumPy/SciPy pipeline, ModernGL 3D rendering, Dear PyGui.

**Start here:** :doc:`user_guide/quickstart` (run the app) and :doc:`user_guide/interface` (panels and controls). Under **Concepts**, :doc:`concepts/overview` sketches signal flow, :doc:`concepts/biology` gives retina context, and :doc:`concepts/equations` states the model in symbols.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   user_guide/quickstart
   user_guide/interface

.. toctree::
   :maxdepth: 2
   :caption: Concepts (theory & equations)

   concepts/overview
   concepts/biology
   concepts/equations

.. toctree::
   :maxdepth: 1
   :caption: Deploy

   user_guide/deploy_github_pages

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
