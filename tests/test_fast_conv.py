"""Tests for fast_conv.gaussian_pool_2d (SciPy path, optional Cython)."""
from __future__ import annotations

import numpy as np

from src.simulation.fast_conv import gaussian_pool_2d


def test_gaussian_pool_2d_shape():
    """Output shape equals input shape."""
    a = np.random.randn(32, 48).astype(np.float32)
    out = gaussian_pool_2d(a, sigma=2.0, mode="reflect")
    assert out.shape == a.shape


def test_gaussian_pool_2d_smooths():
    """Larger sigma smooths more (lower variance of output)."""
    a = np.random.randn(64, 64).astype(np.float32)
    out_small = gaussian_pool_2d(a, sigma=0.5, mode="reflect")
    out_large = gaussian_pool_2d(a, sigma=4.0, mode="reflect")
    assert float(np.var(out_large)) < float(np.var(out_small))


def test_gaussian_pool_2d_constant():
    """Constant input remains constant."""
    a = np.ones((16, 16), dtype=np.float32) * 7.0
    out = gaussian_pool_2d(a, sigma=3.0, mode="reflect")
    np.testing.assert_array_almost_equal(out, 7.0)
