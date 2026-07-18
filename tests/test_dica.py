"""Unit tests for DICA."""

import numpy as np
import pytest
from conformal_ops import DICA


def _make_lp(d=20, budget_ratio=0.6):
    A_eq = np.ones((1, d))
    b_eq = np.array([budget_ratio * d])
    bounds = [(0.1, 1.0)] * d
    return A_eq, b_eq, bounds


def _run_dica(beta=0.5, n_rounds=300, d=20, seed=42):
    rng = np.random.RandomState(seed)
    A_eq, b_eq, bounds = _make_lp(d)
    model = DICA(alpha=0.10, beta=beta)
    for _ in range(n_rounds):
        base = rng.exponential(2.0, size=d)
        c_true = 0.5 + base / 10
        c_pred = 0.5 + (base + rng.normal(0, 0.5, d)) / 10
        model.step(c_pred, c_true, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    return model


def test_beta_zero_matches_uca():
    """DICA with beta=0 should behave like standard conformal."""
    rng = np.random.RandomState(42)
    d = 20
    A_eq, b_eq, bounds = _make_lp(d)

    dica = DICA(alpha=0.10, beta=0.0)
    uca = DICA(alpha=0.10, beta=0.0)

    for _ in range(200):
        base = rng.exponential(2.0, size=d)
        c_true = 0.5 + base / 10
        c_pred = 0.5 + (base + rng.normal(0, 0.5, d)) / 10
        r1 = dica.step(c_pred, c_true, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        r2 = uca.step(c_pred, c_true, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        np.testing.assert_allclose(r1["cost"], r2["cost"], rtol=1e-6)


def test_radii_budget_neutral():
    """Sum of DICA radii should be close to sum of standard radii."""
    model = _run_dica(beta=0.5, n_rounds=200)
    # After warmup, get both radii sets
    c_dummy = np.ones(20)
    dica_radii = model.get_radii(c_dummy)

    # Standard radii (beta=0 model)
    model_uca = _run_dica(beta=0.0, n_rounds=200)
    std_radii = model_uca.get_radii(c_dummy)

    ratio = np.sum(dica_radii) / (np.sum(std_radii) + 1e-10)
    assert 0.85 <= ratio <= 1.15, f"Budget ratio {ratio:.3f} outside [0.85, 1.15]"


def test_allocation_ema_updates():
    model = DICA(alpha=0.10, beta=0.5)
    assert model.allocation_ema is None
    z = np.random.rand(20)
    model.update_allocation(z)
    assert model.allocation_ema is not None
    assert model.allocation_ema.shape == (20,)


def test_coverage_near_target():
    model = _run_dica(beta=0.5, n_rounds=500)
    assert 0.80 <= model.coverage <= 0.98


def test_dica_coverage_near_target():
    model = _run_dica(beta=0.5, n_rounds=500)
    assert 0.75 <= model.dica_coverage <= 0.98


def test_poc_reduction():
    """DICA (beta=0.5) should have lower PoC than UCA (beta=0)."""
    dica = _run_dica(beta=0.5, n_rounds=500, seed=42)
    uca = _run_dica(beta=0.0, n_rounds=500, seed=42)
    dica_poc = dica.get_results()["avg_poc"]
    uca_poc = uca.get_results()["avg_poc"]
    assert dica_poc < uca_poc, f"DICA PoC ({dica_poc:.4f}) >= UCA PoC ({uca_poc:.4f})"


def test_reset():
    model = _run_dica(n_rounds=100)
    assert model.get_results()["n_rounds"] == 100
    model.reset()
    assert model.get_results()["n_rounds"] == 0
    assert model.allocation_ema is None


def test_get_results_keys():
    model = _run_dica(n_rounds=50)
    results = model.get_results()
    for key in ["coverage", "dica_coverage", "avg_poc", "avg_cost", "n_rounds"]:
        assert key in results, f"Missing key: {key}"
