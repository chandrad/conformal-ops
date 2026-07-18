"""Unit tests for ICU allocation problems."""

import numpy as np
import pytest
from conformal_ops.problems import (
    NurseStaffingProblem,
    BedAllocationProblem,
    DischargePlanningProblem,
)


def test_nurse_lp_feasible():
    prob = NurseStaffingProblem(d=20)
    c = np.random.rand(20) + 0.5
    z, cost = prob.solve(c)
    assert z is not None
    assert len(z) == 20


def test_nurse_budget_satisfied():
    prob = NurseStaffingProblem(d=20, budget_ratio=0.6)
    c = np.random.rand(20) + 0.5
    z, _ = prob.solve(c)
    np.testing.assert_allclose(np.sum(z), 0.6 * 20, atol=1e-6)


def test_nurse_bounds_satisfied():
    prob = NurseStaffingProblem(d=20, min_staffing=0.1)
    c = np.random.rand(20) + 0.5
    z, _ = prob.solve(c)
    assert np.all(z >= 0.1 - 1e-8), f"Min bound violated: {z.min()}"
    assert np.all(z <= 1.0 + 1e-8), f"Max bound violated: {z.max()}"


def test_bed_lp_feasible():
    prob = BedAllocationProblem(d=30)
    los_pred = np.random.exponential(2.0, size=30)
    c = np.random.rand(30) + 0.5
    constraints = prob.get_constraints(los_pred)
    z, cost = prob.solve(c, constraints["A_ub"], constraints["b_ub"])
    assert z is not None
    assert len(z) == 30


def test_discharge_lp_feasible():
    prob = DischargePlanningProblem(d=50, budget=15.0)
    c = np.random.rand(50) + 0.5
    z, cost = prob.solve(c)
    assert z is not None
    assert len(z) == 50
    np.testing.assert_allclose(np.sum(z), 15.0, atol=1e-6)


def test_nurse_get_constraints():
    prob = NurseStaffingProblem(d=20)
    cons = prob.get_constraints()
    assert "A_eq" in cons
    assert "bounds" in cons
    assert cons["A_eq"].shape == (1, 20)
    assert len(cons["bounds"]) == 20


def test_build_costs():
    prob = NurseStaffingProblem(d=20, los_scale=336.0)
    los_pred = np.random.exponential(50, size=20)
    los_true = np.random.exponential(50, size=20)
    c_pred, c_true = prob.build_costs(los_pred, los_true)
    assert c_pred.shape == (20,)
    assert np.all(c_pred > 0)
    assert np.all(c_true > 0)
