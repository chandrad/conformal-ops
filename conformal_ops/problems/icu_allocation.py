"""
ICU Resource Allocation Problems for Conformal Predict-then-Optimize.

Three clinical LP formulations:
1. Nurse Staffing (d=20): budget-constrained assignment
2. Bed Allocation (d=30): knapsack with minimum admission
3. Discharge Planning (d=50): budget-constrained follow-up
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple, Optional, Dict, Any


class NurseStaffingProblem:
    """
    Nurse-patient assignment LP.

    Allocate a fixed nursing budget B across d patients, minimizing
    acuity-weighted staffing cost. Each patient gets at least min_staffing.

    Constraints (equality + bounds):
        sum(z_j) = B
        min_staffing <= z_j <= 1

    Parameters
    ----------
    d : int
        Number of patients per shift.
    budget_ratio : float
        Budget as fraction of max staffing (B = budget_ratio * d).
    min_staffing : float
        Minimum allocation per patient.
    los_scale : float
        Fixed LOS normalizer (336h for MIMIC, 14d for diabetes).
    """

    def __init__(self, d: int = 20, budget_ratio: float = 0.6,
                 min_staffing: float = 0.1, los_scale: float = 336.0):
        self.d = d
        self.budget_ratio = budget_ratio
        self.min_staffing = min_staffing
        self.los_scale = los_scale

        self.A_eq = np.ones((1, d))
        self.b_eq = np.array([budget_ratio * d])
        self.bounds = [(min_staffing, 1.0)] * d

    def build_costs(self, los_pred: np.ndarray, los_true: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert LOS to cost vectors."""
        c_true = 0.5 + los_true / self.los_scale
        c_pred = 0.5 + los_pred / self.los_scale
        return c_pred, c_true

    def solve(self, c: np.ndarray) -> Tuple[np.ndarray, float]:
        """Solve LP with given cost vector."""
        res = linprog(c, A_eq=self.A_eq, b_eq=self.b_eq,
                      bounds=self.bounds, method="highs")
        if res.success:
            return res.x, float(c @ res.x)
        z = np.ones(self.d) * self.b_eq[0] / self.d
        return z, float(c @ z)

    def get_constraints(self) -> Dict[str, Any]:
        """Return LP constraints for use with DICA.step()."""
        return {
            "A_eq": self.A_eq, "b_eq": self.b_eq,
            "A_ub": None, "b_ub": None,
            "bounds": self.bounds,
        }


class BedAllocationProblem:
    """
    ICU bed allocation as knapsack LP relaxation.

    Allocate fractional bed-days with capacity and minimum admission
    constraints.

    Constraints (inequality + bounds):
        w^T z <= C  (capacity)
        sum(z_j) >= M  (minimum admissions)
        0 <= z_j <= 1
    """

    def __init__(self, d: int = 30, capacity_ratio: float = 0.5,
                 min_admit_ratio: float = 0.4, los_scale: float = 336.0):
        self.d = d
        self.capacity_ratio = capacity_ratio
        self.min_admit_ratio = min_admit_ratio
        self.los_scale = los_scale
        self.bounds = [(0.0, 1.0)] * d

    def build_costs(self, los_pred: np.ndarray, los_true: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
        c_true = 0.5 + los_true / self.los_scale
        c_pred = 0.5 + los_pred / self.los_scale
        return c_pred, c_true

    def get_constraints(self, los_pred: np.ndarray) -> Dict[str, Any]:
        """Return LP constraints (depend on predicted LOS for weights)."""
        weights = np.maximum(los_pred, 1.0)
        capacity = self.capacity_ratio * weights.sum()
        min_admit = int(self.min_admit_ratio * self.d)

        A_ub = np.vstack([
            weights.reshape(1, -1),
            -np.ones((1, self.d)),
        ])
        b_ub = np.array([capacity, -min_admit])

        return {
            "A_eq": None, "b_eq": None,
            "A_ub": A_ub, "b_ub": b_ub,
            "bounds": self.bounds,
        }

    def solve(self, c: np.ndarray, A_ub: np.ndarray, b_ub: np.ndarray
              ) -> Tuple[np.ndarray, float]:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                      bounds=self.bounds, method="highs")
        if res.success:
            return res.x, float(c @ res.x)
        return np.ones(self.d) / self.d, float(c @ np.ones(self.d) / self.d)


class DischargePlanningProblem:
    """
    Discharge follow-up resource assignment LP.

    Allocate follow-up capacity to patients based on predicted
    readmission risk (proxied by LOS).

    Constraints (equality + bounds):
        sum(z_j) = budget
        0 <= z_j <= 1
    """

    def __init__(self, d: int = 50, budget: float = 15.0,
                 los_scale: float = 336.0):
        self.d = d
        self.budget = budget
        self.los_scale = los_scale

        self.A_eq = np.ones((1, d))
        self.b_eq = np.array([budget])
        self.bounds = [(0.0, 1.0)] * d

    def build_costs(self, los_pred: np.ndarray, los_true: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
        c_true = 0.5 + los_true / self.los_scale
        c_pred = 0.5 + los_pred / self.los_scale
        return c_pred, c_true

    def solve(self, c: np.ndarray) -> Tuple[np.ndarray, float]:
        res = linprog(c, A_eq=self.A_eq, b_eq=self.b_eq,
                      bounds=self.bounds, method="highs")
        if res.success:
            return res.x, float(c @ res.x)
        z = np.ones(self.d) * self.budget / self.d
        return z, float(c @ z)

    def get_constraints(self) -> Dict[str, Any]:
        return {
            "A_eq": self.A_eq, "b_eq": self.b_eq,
            "A_ub": None, "b_ub": None,
            "bounds": self.bounds,
        }
