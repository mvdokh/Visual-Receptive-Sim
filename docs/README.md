# RGC Circuit Simulator — Documentation

Read the Docs–style Sphinx documentation. Deployable to **GitHub Pages**.

## Build locally

```bash
pip install -r docs/requirements.txt
cd docs && sphinx-build -b html . _build/html
```

Open `_build/html/index.html` in a browser.

## How to add the docs site on GitHub

1. **Push** this repo to GitHub (including the `docs/` folder and `.github/workflows/docs.yml`).

2. In the repo on GitHub, go to **Settings** → **Pages**.

3. Under **Build and deployment**, set **Source** to **GitHub Actions** (not “Deploy from a branch”).

4. Run the **Docs** workflow once: **Actions** → **Docs** → **Run workflow** (or push a commit to `main`/`master`).

5. After it finishes, the site is live at:
   **`https://<your-username>.github.io/<repo-name>/`**  
   (e.g. `https://martindokholyan.github.io/Visual-Receptive-Sim/`).

No need to create a `gh-pages` branch manually; the workflow builds and deploys the HTML for you.
