"""
Baseline methods for benchmarking against DICA.

All methods share the same `.step()` interface as DICA for easy comparison.

Methods:
    UCA     — Uniform Conformal Allocation (DICA with beta=0)
    CPO     — Conformal Predict-then-Optimize (split conformal, periodic recalibration)
    EWMA    — Exponential Weighted Moving Average (ad-hoc heuristic, no coverage target)
    ACRO    — Allocation-Conditional Radii Optimization (group-conditional conformal)
    Nominal — Solve LP with predictions directly (no robustification)
    FixedMargin — Add fixed percentage buffer to predictions

Reference:
    Dronavajjala (2026). Decision-Informed Online Conformal Prediction
    for ICU Resource Allocation. PMLR 340, MLHC 2026.
"""

import numpy as np
from scipy.optimize import linprog
from typing import Optional, Dict, List

from conformal_ops.dica.dica import DICA
from conformal_ops.core.online_conformal import OnlineConformal


class UCA(DICA):
    """
    Uniform Conformal Allocation — standard online conformal with uniform radii.

    Equivalent to DICA with beta=0 (no redistribution).
    Serves as the primary baseline: same coverage, higher PoC.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate.
    eta : float
        Gibbs-Candes step size.
    window : int
        Score history window.
    """

    def __init__(self, alpha: float = 0.10, eta: float = 0.05,
                 window: int = 150, scale_ema: float = 0.05):
        super().__init__(alpha=alpha, eta=eta, window=window,
                         beta=0.0, scale_ema=scale_ema)


class CPO:
    """
    Conformal Predict-then-Optimize (Patel et al., AISTATS 2024).

    Split conformal with periodic recalibration. Uses a held-out
    calibration set of recent scores, recomputing the quantile every
    `recalib_freq` rounds. Under distribution shift, the calibration
    set becomes stale between windows, causing coverage degradation.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate.
    recalib_freq : int
        Rounds between recalibration.
    """

    def __init__(self, alpha: float = 0.10, recalib_freq: int = 50):
        self.alpha = alpha
        self.recalib_freq = recalib_freq
        self._calib_scores: List[float] = []
        self._q: float = 1.0
        self._sigma: Optional[np.ndarray] = None
        self._residuals: List[np.ndarray] = []
        self._costs: List[float] = []
        self._nominal_costs: List[float] = []
        self._coverages: List[float] = []
        self._step: int = 0

    @property
    def coverage(self) -> float:
        return float(np.mean(self._coverages)) if self._coverages else 0.0

    @property
    def poc_history(self) -> List[float]:
        return [(rc / nc - 1) if nc > 1e-10 else 0.0
                for rc, nc in zip(self._costs, self._nominal_costs)]

    def step(self, c_pred, c_true, A_eq=None, b_eq=None,
             A_ub=None, b_ub=None, bounds=None) -> Dict:
        d = len(c_pred)

        # Recalibrate periodically
        if self._step > 0 and self._step % self.recalib_freq == 0:
            if self._calib_scores:
                idx = min(int(np.ceil((1 - self.alpha) * len(self._calib_scores))),
                          len(self._calib_scores) - 1)
                self._q = np.sort(self._calib_scores)[idx]

        # Compute radii
        if self._sigma is None:
            self._sigma = np.ones(d) * 0.1
        radii = self._q * self._sigma

        # Robust LP
        c_robust = c_pred + radii
        res = linprog(c_robust, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                      bounds=bounds, method="highs")
        z_opt = res.x if res.success else np.ones(d) / d

        cost = float(c_true @ z_opt)
        self._costs.append(cost)

        # Nominal cost
        res_nom = linprog(c_pred, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                          bounds=bounds, method="highs")
        nominal_cost = float(c_true @ res_nom.x) if res_nom.success else cost
        self._nominal_costs.append(nominal_cost)

        # Update calibration
        residual = c_true - c_pred
        self._residuals.append(residual)
        if len(self._residuals) > 200:
            self._residuals = self._residuals[-200:]
        if len(self._residuals) >= 5:
            R = np.array(self._residuals)
            self._sigma = np.maximum(np.std(R, axis=0), 1e-6)

        score = np.max(np.abs(residual) / self._sigma)
        self._calib_scores.append(score)
        # Window calibration scores to match residual window (prevents staleness)
        if len(self._calib_scores) > 200:
            self._calib_scores = self._calib_scores[-200:]
        covered = float(np.all(np.abs(residual) <= radii + 1e-10))
        self._coverages.append(covered)
        self._step += 1

        poc = (cost / nominal_cost - 1) if nominal_cost > 1e-10 else 0.0
        return {"z_opt": z_opt, "cost": cost, "nominal_cost": nominal_cost,
                "radii": radii, "std_covered": covered, "poc": poc}

    def get_results(self) -> Dict:
        avg_poc = float(np.mean(self.poc_history)) if self.poc_history else 0.0
        return {"coverage": self.coverage, "avg_poc": avg_poc,
                "avg_cost": float(np.mean(self._costs)) if self._costs else 0.0,
                "n_rounds": len(self._costs)}

    def reset(self):
        self._calib_scores = []
        self._q = 1.0
        self._sigma = None
        self._residuals = []
        self._costs = []
        self._nominal_costs = []
        self._coverages = []
        self._step = 0


class EWMA:
    """
    Exponential Weighted Moving Average baseline.

    Uses EWMA of recent costs + fixed multiplier for robustification.
    Does NOT target a specific coverage level. Included to illustrate
    the gap between ad-hoc heuristics and calibrated conformal methods.

    Parameters
    ----------
    gamma : float
        Multiplier for EWMA standard deviation.
    decay : float
        EMA decay rate.
    """

    def __init__(self, gamma: float = 1.5, decay: float = 0.05):
        self.gamma = gamma
        self.decay = decay
        self._mu: Optional[np.ndarray] = None
        self._var: Optional[np.ndarray] = None
        self._costs: List[float] = []
        self._nominal_costs: List[float] = []
        self._coverages: List[float] = []

    @property
    def coverage(self) -> float:
        return float(np.mean(self._coverages)) if self._coverages else 0.0

    @property
    def poc_history(self) -> List[float]:
        return [(rc / nc - 1) if nc > 1e-10 else 0.0
                for rc, nc in zip(self._costs, self._nominal_costs)]

    def step(self, c_pred, c_true, A_eq=None, b_eq=None,
             A_ub=None, b_ub=None, bounds=None) -> Dict:
        d = len(c_pred)
        if self._mu is None:
            self._mu = c_pred.copy()
            self._var = np.ones(d) * 0.01

        radii = self.gamma * np.sqrt(self._var)
        # Use c_pred (same as other methods) for fair comparison
        c_robust = c_pred + radii

        res = linprog(c_robust, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                      bounds=bounds, method="highs")
        z_opt = res.x if res.success else np.ones(d) / d

        cost = float(c_true @ z_opt)
        self._costs.append(cost)

        res_nom = linprog(c_pred, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                          bounds=bounds, method="highs")
        nominal_cost = float(c_true @ res_nom.x) if res_nom.success else cost
        self._nominal_costs.append(nominal_cost)

        # Coverage: check if true costs fall within c_pred ± radii
        covered = float(np.all(np.abs(c_true - c_pred) <= radii + 1e-10))
        self._coverages.append(covered)

        # Update EWMA
        old_mu = self._mu.copy()
        self._var = (1 - self.decay) * self._var + self.decay * (c_true - old_mu) ** 2
        self._mu = (1 - self.decay) * self._mu + self.decay * c_true

        poc = (cost / nominal_cost - 1) if nominal_cost > 1e-10 else 0.0
        return {"z_opt": z_opt, "cost": cost, "nominal_cost": nominal_cost,
                "radii": radii, "std_covered": covered, "poc": poc}

    def get_results(self) -> Dict:
        avg_poc = float(np.mean(self.poc_history)) if self.poc_history else 0.0
        return {"coverage": self.coverage, "avg_poc": avg_poc,
                "avg_cost": float(np.mean(self._costs)) if self._costs else 0.0,
                "n_rounds": len(self._costs)}

    def reset(self):
        self._mu = None
        self._var = None
        self._costs = []
        self._nominal_costs = []
        self._coverages = []


class ACRO:
    """
    Allocation-Conditional Radii Optimization.

    Maintains separate conformal quantiles per patient acuity group
    (low/medium/high LOS). Tests whether group-level calibration
    reduces PoC compared to DICA's allocation-feedback mechanism.

    Uses Mondrian-style group-conditional conformal prediction.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate.
    n_groups : int
        Number of acuity strata.
    """

    def __init__(self, alpha: float = 0.10, eta: float = 0.05,
                 window: int = 150, n_groups: int = 3):
        self.alpha_target = alpha
        self.eta = eta
        self.window = window
        self.n_groups = n_groups
        # Per-group state: scalar scores, alpha, sigma (as list of per-patient values)
        self._group_scores: Dict[int, List[float]] = {g: [] for g in range(n_groups)}
        self._group_alpha: Dict[int, float] = {g: alpha for g in range(n_groups)}
        self._group_sigma: Dict[int, List[float]] = {g: [] for g in range(n_groups)}
        self._scale_ema = 0.05
        self._costs: List[float] = []
        self._nominal_costs: List[float] = []
        self._coverages: List[float] = []

    @property
    def coverage(self) -> float:
        return float(np.mean(self._coverages)) if self._coverages else 0.0

    @property
    def poc_history(self) -> List[float]:
        return [(rc / nc - 1) if nc > 1e-10 else 0.0
                for rc, nc in zip(self._costs, self._nominal_costs)]

    def _assign_groups(self, c_pred: np.ndarray) -> np.ndarray:
        """Assign patients to acuity groups by prediction terciles."""
        thresholds = np.quantile(c_pred, [1/3, 2/3])
        groups = np.zeros(len(c_pred), dtype=int)
        groups[c_pred > thresholds[0]] = 1
        groups[c_pred > thresholds[1]] = 2
        return groups

    def _get_group_q(self, g: int) -> float:
        """Get quantile for group g from its score history."""
        scores = self._group_scores[g]
        if len(scores) < 5:
            return 1.0
        level = min(1 - self._group_alpha[g], 1.0)
        return float(np.quantile(scores, level))

    def step(self, c_pred, c_true, A_eq=None, b_eq=None,
             A_ub=None, b_ub=None, bounds=None) -> Dict:
        d = len(c_pred)
        groups = self._assign_groups(c_pred)

        # Get per-patient radii from group quantiles
        radii = np.zeros(d)
        for j in range(d):
            g = groups[j]
            q_g = self._get_group_q(g)
            # Use per-patient sigma if available, else fallback
            if j < len(self._group_sigma.get(g, [])):
                sigma_j = self._group_sigma[g][j] if self._group_sigma[g] else 0.3
            else:
                sigma_j = 0.3  # fallback before calibration
            # Use global sigma approach: track per-group average sigma
            if self._group_scores[g]:
                # Use mean of recent residuals for this group as sigma proxy
                sigma_j = max(np.mean([abs(s) for s in self._group_scores[g][-20:]]) * 0.5, 0.01)
            radii[j] = q_g * sigma_j

        # Robust LP
        c_robust = c_pred + radii
        res = linprog(c_robust, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                      bounds=bounds, method="highs")
        z_opt = res.x if res.success else np.ones(d) / d

        cost = float(c_true @ z_opt)
        self._costs.append(cost)

        res_nom = linprog(c_pred, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                          bounds=bounds, method="highs")
        nominal_cost = float(c_true @ res_nom.x) if res_nom.success else cost
        self._nominal_costs.append(nominal_cost)

        # Update each group with SCALAR max-residual score (not subvectors)
        all_covered = True
        for g in range(self.n_groups):
            mask = groups == g
            if not mask.any():
                continue
            residuals_g = np.abs(c_true[mask] - c_pred[mask])
            # Scalar score for this group: max normalized residual
            if self._group_scores[g]:
                sigma_g = max(np.mean([abs(s) for s in self._group_scores[g][-20:]]) * 0.5, 0.01)
            else:
                sigma_g = max(np.mean(residuals_g), 0.01)
            score_g = float(np.max(residuals_g / sigma_g))
            self._group_scores[g].append(score_g)
            if len(self._group_scores[g]) > self.window:
                self._group_scores[g] = self._group_scores[g][-self.window:]

            # Coverage check for this group
            q_g = self._get_group_q(g)
            if score_g > q_g:
                all_covered = False

            # Update group alpha (Gibbs-Candes per group)
            err_g = 1.0 - float(score_g <= q_g)
            self._group_alpha[g] = float(np.clip(
                self._group_alpha[g] + self.eta * (self.alpha_target - err_g),
                0.01, 0.5
            ))

        self._coverages.append(float(all_covered))
        poc = (cost / nominal_cost - 1) if nominal_cost > 1e-10 else 0.0
        return {"z_opt": z_opt, "cost": cost, "nominal_cost": nominal_cost,
                "radii": radii, "std_covered": float(all_covered), "poc": poc}

    def get_results(self) -> Dict:
        avg_poc = float(np.mean(self.poc_history)) if self.poc_history else 0.0
        return {"coverage": self.coverage, "avg_poc": avg_poc,
                "avg_cost": float(np.mean(self._costs)) if self._costs else 0.0,
                "n_rounds": len(self._costs)}

    def reset(self):
        self._group_scores = {g: [] for g in range(self.n_groups)}
        self._group_alpha = {g: self.alpha_target for g in range(self.n_groups)}
        self._group_sigma = {g: [] for g in range(self.n_groups)}
        self._costs = []
        self._nominal_costs = []
        self._coverages = []


class Nominal:
    """
    Nominal baseline — solve LP with predictions directly, no robustification.

    Always has PoC = 0 by definition (it's the denominator).
    Coverage is undefined (no conformal sets constructed).
    """

    def __init__(self):
        self._costs: List[float] = []

    def step(self, c_pred, c_true, A_eq=None, b_eq=None,
             A_ub=None, b_ub=None, bounds=None) -> Dict:
        res = linprog(c_pred, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                      bounds=bounds, method="highs")
        z_opt = res.x if res.success else np.ones(len(c_pred)) / len(c_pred)
        cost = float(c_true @ z_opt)
        self._costs.append(cost)
        return {"z_opt": z_opt, "cost": cost, "nominal_cost": cost,
                "radii": np.zeros(len(c_pred)), "std_covered": 0.0, "poc": 0.0}

    def get_results(self) -> Dict:
        return {"coverage": 0.0, "avg_poc": 0.0,
                "avg_cost": float(np.mean(self._costs)) if self._costs else 0.0,
                "n_rounds": len(self._costs)}

    def reset(self):
        self._costs = []


class FixedMargin:
    """
    Fixed margin baseline — add fixed percentage buffer to predictions.

    A common operational heuristic (e.g., "add 10% to all estimates").

    Parameters
    ----------
    margin : float
        Buffer as fraction of predicted cost (default 0.10 = 10%).
    """

    def __init__(self, margin: float = 0.10):
        self.margin = margin
        self._costs: List[float] = []
        self._nominal_costs: List[float] = []
        self._coverages: List[float] = []

    @property
    def coverage(self) -> float:
        return float(np.mean(self._coverages)) if self._coverages else 0.0

    @property
    def poc_history(self) -> List[float]:
        return [(rc / nc - 1) if nc > 1e-10 else 0.0
                for rc, nc in zip(self._costs, self._nominal_costs)]

    def step(self, c_pred, c_true, A_eq=None, b_eq=None,
             A_ub=None, b_ub=None, bounds=None) -> Dict:
        radii = np.abs(c_pred) * self.margin
        c_robust = c_pred + radii

        res = linprog(c_robust, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                      bounds=bounds, method="highs")
        z_opt = res.x if res.success else np.ones(len(c_pred)) / len(c_pred)

        cost = float(c_true @ z_opt)
        self._costs.append(cost)

        res_nom = linprog(c_pred, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                          bounds=bounds, method="highs")
        nominal_cost = float(c_true @ res_nom.x) if res_nom.success else cost
        self._nominal_costs.append(nominal_cost)

        covered = float(np.all(np.abs(c_true - c_pred) <= radii + 1e-10))
        self._coverages.append(covered)

        poc = (cost / nominal_cost - 1) if nominal_cost > 1e-10 else 0.0
        return {"z_opt": z_opt, "cost": cost, "nominal_cost": nominal_cost,
                "radii": radii, "std_covered": covered, "poc": poc}

    def get_results(self) -> Dict:
        avg_poc = float(np.mean(self.poc_history)) if self.poc_history else 0.0
        return {"coverage": self.coverage, "avg_poc": avg_poc,
                "avg_cost": float(np.mean(self._costs)) if self._costs else 0.0,
                "n_rounds": len(self._costs)}

    def reset(self):
        self._costs = []
        self._nominal_costs = []
        self._coverages = []
