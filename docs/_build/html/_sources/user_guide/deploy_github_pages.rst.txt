Documentation site
==================

The RGC Circuit Simulator documentation is built with **Sphinx** using the
Read the Docs theme. As a user of the simulator you don't need to build or
deploy the docs: the HTML site is already hosted for you.

If you're viewing the source tree locally and want to read the docs offline,
build them once with:

.. code-block:: bash

   pip install -r docs/requirements.txt
   cd docs
   sphinx-build -b html . _build/html

Then open ``_build/html/index.html`` in your browser.
