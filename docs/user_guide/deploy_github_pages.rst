Deploy documentation to GitHub Pages
===================================

This site is built with **Sphinx** and the **Read the Docs theme**, and can be deployed to **GitHub Pages** so it is available at ``https://<username>.github.io/<repo>/`` (or your custom domain).

Option A: Build and publish from your machine
---------------------------------------------

1. **Install docs dependencies**

   .. code-block:: bash

      pip install sphinx sphinx-rtd-theme

2. **Build the HTML docs**

   From the project root:

   .. code-block:: bash

      cd docs
      sphinx-build -b html . _build/html

   Output is in ``docs/_build/html``.

3. **Push to a branch GitHub uses for Pages**

   - On GitHub, go to the repo → **Settings** → **Pages**.
   - Under **Source**, choose **Deploy from a branch**.
   - Branch: pick **gh-pages** (or **main** and folder **/docs** if you use that setup). If you use **gh-pages**, the branch must contain the *contents* of ``_build/html`` (the HTML files at the root of the branch), not the whole repo.
   - Typical workflow: clone the repo, build as above, then push *only* the contents of ``_build/html`` to the **gh-pages** branch (e.g. with a separate clone or ``git subtree``). See `GitHub Pages docs <https://docs.github.com/en/pages>`_ for “publishing from a branch”.

Option B: GitHub Actions (recommended)
--------------------------------------

The project includes a workflow that builds Sphinx and publishes to GitHub Pages so you don’t build locally.

1. **Enable GitHub Pages in the repo**

   - Open the repo on GitHub → **Settings** → **Pages**.
   - Under **Build and deployment**, set **Source** to **GitHub Actions** (not “Deploy from a branch”).

2. **Push the workflow and docs**

   - Ensure the ``.github/workflows/docs.yml`` workflow (and the ``docs/`` folder) is in your repo and push to the default branch (e.g. ``main``).

3. **First run**

   - Go to **Actions** → select the “Docs” workflow → run it (or push a commit). When it finishes, the site will be available at:

     **https://\<username\>.github.io/\<repo\>/**  

     (e.g. ``https://martindokholyan.github.io/Visual-Receptive-Sim/`` if the repo is ``Visual-Receptive-Sim`` under user ``martindokholyan``).

4. **Subpath**

   If the site is served under a subpath (e.g. ``/Visual-Receptive-Sim/``), the workflow sets ``html_baseurl`` so links work correctly. No extra steps are needed once the workflow and ``conf.py`` use that base URL.

How to add the site when you go to GitHub
-----------------------------------------

- **If you already pushed the repo**: Enable Pages (Settings → Pages → Source: **GitHub Actions**), then run the **Docs** workflow once from the Actions tab. The site URL will appear under Settings → Pages after the first successful run.
- **If you haven’t pushed yet**: Push the repo (including ``docs/`` and ``.github/workflows/docs.yml``), then do the same: Settings → Pages → Source: **GitHub Actions**, and run the Docs workflow.

To use a **custom domain**, set it under Settings → Pages → Custom domain and add the DNS records GitHub specifies.
