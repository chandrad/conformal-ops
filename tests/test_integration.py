"""End-to-end integration test."""

import numpy as np
from conformal_ops import DICA
from conformal_ops.problems import NurseStaffingProblem


def test_full_pipeline():
    """Run 200 rounds of DICA on nurse staffing, verify basic sanity."""
    rng = np.random.RandomState(42)
    prob = NurseStaffingProblem(d=20, los_scale=10.0)
    cons = prob.get_constraints()

    dica = DICA(alpha=0.10, beta=0.5)

    for t in range(200):
        los_true = rng.exponential(2.0, size=20)
        los_pred = los_true + rng.normal(0, 0.5, size=20)
        c_pred, c_true = prob.build_costs(los_pred, los_true)

        result = dica.step(c_pred, c_true, **cons)

        assert "z_opt" in result
        assert "cost" in result
        assert "poc" in result
        assert len(result["z_opt"]) == 20

    stats = dica.get_results()
    assert stats["n_rounds"] == 200
    assert stats["coverage"] > 0.70
    assert stats["avg_poc"] > -0.5  # PoC should be reasonable


def test_dica_vs_uca_integration():
    """DICA should have lower PoC than UCA on the same data."""
    rng_base = np.random.RandomState(123)
    data = []
    for _ in range(300):
        los_true = rng_base.exponential(2.0, size=20)
        los_pred = los_true + rng_base.normal(0, 0.5, size=20)
        data.append((los_pred, los_true))

    prob = NurseStaffingProblem(d=20, los_scale=10.0)
    cons = prob.get_constraints()

    results = {}
    for name, beta in [("UCA", 0.0), ("DICA", 0.5)]:
        model = DICA(alpha=0.10, beta=beta)
        for los_pred, los_true in data:
            c_pred, c_true = prob.build_costs(los_pred, los_true)
            model.step(c_pred, c_true, **cons)
        results[name] = model.get_results()

    assert results["DICA"]["avg_poc"] < results["UCA"]["avg_poc"]
