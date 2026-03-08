"""
Build Cython extensions. Run: python setup.py build_ext --inplace
"""
from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "convolve_2d",
        ["convolve_2d.pyx"],
        include_dirs=[],
    ),
    Extension(
        "rf_probe_sweep",
        ["rf_probe_sweep.pyx"],
        include_dirs=[],
    ),
    Extension(
        "layer_update",
        ["layer_update.pyx"],
        include_dirs=[],
    ),
]

setup(
    name="hot_numerical",
    ext_modules=cythonize(extensions),
)
