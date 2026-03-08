# Configuration file for the Sphinx documentation builder.
# RGC Circuit Simulator — Read the Docs style, deployable to GitHub Pages.

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "RGC Circuit Simulator"
copyright = "RGC Circuit Simulator contributors"
author = "RGC Circuit Simulator"
release = "0.1"
version = "0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_static_path = ["_static"]

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "titles_only": False,
}

html_title = "RGC Circuit Simulator"
html_short_title = "RGC Sim"
html_logo = None
html_favicon = None

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
mathjax3_config = {
    "tex": {
        "inlineMath": [["\\(", "\\)"]],
        "displayMath": [["\\[", "\\]"]],
    }
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

nitpicky = False

# GitHub Pages: set base URL when deploying to username.github.io/repo/
_baseurl = os.environ.get("SPHINX_HTML_BASEURL", "")
if _baseurl:
    html_baseurl = _baseurl.rstrip("/") + "/"
