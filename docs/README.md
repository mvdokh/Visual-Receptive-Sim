# RGC Circuit Simulator — Documentation

Read the Docs–style **Sphinx** documentation for the simulator. The docs are
purely Python/Sphinx; they do **not** require the Cython extensions to be
built (those only affect the simulation speed, not the docs).

## Build locally

From the project root:

```bash
pip install -r docs/requirements.txt
cd docs
sphinx-build -b html . _build/html
```

Then open `_build/html/index.html` in a browser.

If you have a virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r docs/requirements.txt
cd docs && sphinx-build -b html . _build/html
```

## Docs + Cython

The documentation pages describe the retina model and the pipeline, including
the optional Cython accelerations in `hot_numerical/`, but they **do not**
depend on Cython at build time. You can:

- Build the docs without Cython at all (recommended).
- Optionally build the Cython extensions for the simulator using:

  ```bash
  pip install cython setuptools
  cd hot_numerical
  python setup.py build_ext --inplace
  ```

  (This only affects simulation speed when you run `main.py`, not the docs.)

## How to add the docs site on GitHub

1. **Push** this repo to GitHub (including the `docs/` folder and `.github/workflows/docs.yml`).

2. In the repo on GitHub, go to **Settings** → **Pages**.

3. Under **Build and deployment**, set **Source** to **GitHub Actions** (not “Deploy from a branch”).

4. Run the **Docs** workflow once: **Actions** → **Docs** → **Run workflow** (or push a commit to `main`/`master`).

5. After it finishes, the site is live at:
   **`https://<your-username>.github.io/<repo-name>/`**  
   (e.g. `https://martindokholyan.github.io/Visual-Receptive-Sim/`).

No need to create a `gh-pages` branch manually; the workflow builds and deploys the HTML for you.
