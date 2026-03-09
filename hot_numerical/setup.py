"""
Build Cython extensions. Run: python setup.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys


if sys.platform == "win32":
    extra_compile_args = ["/O2"]
else:
    extra_compile_args = ["-O2", "-ffast-math"]


extensions = [
    # NOTE: module names are *unqualified* so that when we run this setup.py
    # from inside the hot_numerical package directory, the compiled modules
    # land directly next to __init__.py (e.g. convolve_2d.so). Because this
    # directory itself is the hot_numerical package, `import hot_numerical.convolve_2d`
    # still finds these modules correctly.
    Extension(
        "convolve_2d",
        ["convolve_2d.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "rf_probe_sweep",
        ["rf_probe_sweep.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "layer_update",
        ["layer_update.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "stimulus_fill",
        ["stimulus_fill.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="hot_numerical",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
        },
    ),
)
