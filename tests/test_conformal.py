"""Unit tests for OnlineConformal."""

import numpy as np
import pytest
from conformal_ops.core.online_conformal import OnlineConformal


def _run_conformal(n_rounds=500, d=20, seed=42):
    rng = np.random.RandomState(seed)
    model = OnlineConformal(alpha=0.10, eta=0.05, window=150)
    for _ in range(n_rounds):
        base = rng.exponential(2.0, size=d)
        c_true = 0.5 + base / 10
        c_pred = 0.5 + (base + rng.normal(0, 0.5, d)) / 10
        model.update(c_pred, c_true)
    return model


def test_coverage_tracking():
    model = _run_conformal(500)
    cov = model.coverage
    assert 0.80 <= cov <= 0.98, f"Coverage {cov:.3f} outside [0.80, 0.98]"


def test_quantile_positive():
    model = _run_conformal(100)
    assert model.quantile > 0, "Quantile should be positive"


def test_radii_shape():
    model = _run_conformal(50)
    radii = model.get_radii(np.ones(20))
    assert radii.shape == (20,), f"Expected shape (20,), got {radii.shape}"
    assert np.all(radii >= 0), "Radii should be non-negative"


def test_reset():
    model = _run_conformal(100)
    assert model.coverage > 0
    model.reset()
    assert model.coverage == 0.0
    assert model.quantile == 1.0
    assert len(model._scores) == 0


def test_stats():
    model = _run_conformal(100)
    stats = model.get_stats()
    assert "alpha" in stats
    assert "quantile" in stats
    assert "coverage" in stats
    assert 0.01 <= stats["alpha"] <= 0.5
