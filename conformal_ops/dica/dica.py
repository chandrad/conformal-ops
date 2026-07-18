"""
DICA: Decision-Informed Conformal Adaptation.

Uses the downstream optimization's LP allocation as feedback to reshape
conformal radii: tighter where allocation is minimal (saving cost),
wider where allocation is high (strengthening protection).

The Gibbs-Candes quantile update remains standard — only radii are reshaped.

Reference:
    Dronavajjala (2026). Decision-Informed Online Conformal Prediction
    for ICU Resource Allocation. PMLR 340, MLHC 2026.
"""

import numpy as np
from scipy.optimize import linprog
from typing import Optional, Dict, List, Tuple

from conformal_ops.dica.radii import redistribute_radii


class DICA:
    """
    Decision-Informed Conformal Adaptation.

    Maintains online conformal prediction (Gibbs-Candes) with a feedback
    loop: LP allocation -> EMA -> radii redistribution -> next LP.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate (default 0.10 for 90% coverage).
    eta : float
        Step size for Gibbs-Candes alpha update.
    window : int
        Number of recent scores for quantile estimation.
    beta : float
        Redistribution strength. 0 = standard conformal (UCA),
        1 = fully allocation-proportional. Default 0.5.
    allocation_ema_decay : float
        EMA decay for tracking LP allocation. Smaller = smoother.
    scale_ema : float
        EMA decay for per-component noise scale sigma_j.

    Examples
    --------
    >>> from conformal_ops import DICA
    >>> dica = DICA(alpha=0.10, beta=0.5)
    >>> for t in range(T):
    ...     result = dica.step(c_pred, c_true, A_eq=A, b_eq=b, bounds=bnds)
    ...     z_opt = result["z_opt"]
    ...     cost = result["cost"]
    """

    def __init__(
        self,
        alpha: float = 0.10,
        eta: float = 0.05,
        window: int = 150,
        beta: float = 0.5,
        allocation_ema_decay: float = 0.1,
        scale_ema: float = 0.05,
    ):
        self.alpha_target = alpha
        self.alpha = alpha
        self.eta = eta
        self.window = window
        self.beta = beta
        self.allocation_ema_decay = allocation_ema_decay
        self.scale_ema = scale_ema

        self._scores: List[float] = []
        self._sigma: Optional[np.ndarray] = None
        self._allocation_ema: Optional[np.ndarray] = None

        self._std_coverages: List[float] = []
        self._dica_coverages: List[float] = []
        self._costs: List[float] = []
        self._nominal_costs: List[float] = []

    # ── Properties ──────────────────────────────────────────────────

    @property
    def coverage(self) -> float:
        """Scalar coverage (Gibbs-Candes tracked)."""
        return float(np.mean(self._std_coverages)) if self._std_coverages else 0.0

    @property
    def dica_coverage(self) -> float:
        """Coverage under reshaped DICA radii (empirical, no formal guarantee)."""
        return float(np.mean(self._dica_coverages)) if self._dica_coverages else 0.0

    @property
    def allocation_ema(self) -> Optional[np.ndarray]:
        """Current allocation EMA (None before first LP solve)."""
        return self._allocation_ema

    @property
    def poc_history(self) -> List[float]:
        """Per-round Price of Coverage: (robust_cost / nominal_cost) - 1."""
        pocs = []
        for rc, nc in zip(self._costs, self._nominal_costs):
            if nc > 1e-10:
                pocs.append(rc / nc - 1)
            else:
                pocs.append(0.0)
        return pocs

    # ── Core methods ────────────────────────────────────────────────

    def _get_q(self) -> float:
        if len(self._scores) < 5:
            return 1.0
        level = min(1 - self.alpha, 1.0)
        return float(np.quantile(self._scores, level))

    def get_radii(self, c_pred: np.ndarray) -> np.ndarray:
        """
        Compute DICA-reshaped conformal radii.

        Returns standard radii if no allocation history is available yet
        (first round), or reshaped radii based on allocation EMA.

        Parameters
        ----------
        c_pred : array of shape (d,)
            Predicted cost vector.

        Returns
        -------
        radii : array of shape (d,)
            DICA radii (reshaped if allocation history exists).
        """
        d = len(c_pred)
        q = self._get_q()

        if self._sigma is None:
            return np.abs(c_pred) * 0.3

        r_std = np.maximum(q * self._sigma[:d], 0.0)

        if self._allocation_ema is None:
            return r_std

        return redistribute_radii(r_std, self._allocation_ema[:d], self.beta)

    def update(self, c_pred: np.ndarray, c_true: np.ndarray) -> Dict[str, float]:
        """
        Observe true costs, compute score, update quantile.

        Timing (critical): score and DICA coverage use pre-update sigma,
        then sigma is updated via EMA.

        Parameters
        ----------
        c_pred : array of shape (d,)
        c_true : array of shape (d,)

        Returns
        -------
        stats : dict with keys "std_covered", "dica_covered", "score"
        """
        residuals = np.abs(c_true - c_pred)
        d = len(c_pred)

        if self._sigma is None:
            # Compute DICA radii BEFORE initializing sigma (no look-ahead)
            dica_radii = self.get_radii(c_pred)  # returns fallback radii
            dica_covered = float(np.all(residuals[:d] <= dica_radii))
            self._sigma = residuals + 1e-6
            score = 1.0
        else:
            if len(self._sigma) != d:
                dica_radii = self.get_radii(c_pred)
                dica_covered = float(np.all(residuals[:d] <= dica_radii))
                self._sigma = residuals + 1e-6
                score = 1.0
            else:
                # Score uses pre-update sigma
                score = float(np.max(residuals / self._sigma))
                # DICA coverage uses pre-update sigma
                dica_radii = self.get_radii(c_pred)
                dica_covered = float(np.all(residuals[:d] <= dica_radii))
                # NOW update sigma
                self._sigma = (
                    (1 - self.scale_ema) * self._sigma
                    + self.scale_ema * residuals
                )
                self._sigma = np.maximum(self._sigma, 1e-6)

        self._scores.append(score)
        if len(self._scores) > self.window:
            self._scores = self._scores[-self.window:]

        q = self._get_q()
        std_covered = float(score <= q)

        self._std_coverages.append(std_covered)
        self._dica_coverages.append(dica_covered)

        err = 1.0 - std_covered
        self.alpha = float(np.clip(
            self.alpha + self.eta * (self.alpha_target - err), 0.01, 0.5
        ))

        return {
            "std_covered": std_covered,
            "dica_covered": dica_covered,
            "score": score,
        }

    def update_allocation(self, z_opt: np.ndarray):
        """
        Feed LP solution back to update allocation EMA.

        Call this after each LP solve.

        Parameters
        ----------
        z_opt : array of shape (d,)
            LP optimal solution.
        """
        z_abs = np.abs(z_opt)
        if self._allocation_ema is None:
            self._allocation_ema = z_abs.copy()
        else:
            d = len(z_abs)
            if len(self._allocation_ema) != d:
                self._allocation_ema = z_abs.copy()
            else:
                self._allocation_ema = (
                    (1 - self.allocation_ema_decay) * self._allocation_ema
                    + self.allocation_ema_decay * z_abs
                )

    def solve_robust_lp(
        self,
        c_pred: np.ndarray,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
        A_ub: Optional[np.ndarray] = None,
        b_ub: Optional[np.ndarray] = None,
        bounds: Optional[list] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get DICA radii and solve the robust LP.

        Parameters
        ----------
        c_pred : array of shape (d,)
            Predicted cost vector.
        A_eq, b_eq : equality constraints (optional)
        A_ub, b_ub : inequality constraints (optional)
        bounds : list of (lo, hi) tuples (optional)

        Returns
        -------
        z_opt : array of shape (d,)
            Robust LP solution.
        radii : array of shape (d,)
            DICA radii used.
        """
        radii = self.get_radii(c_pred)
        c_robust = c_pred + radii

        res = linprog(
            c_robust, A_eq=A_eq, b_eq=b_eq,
            A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs"
        )

        if res.success:
            return res.x, radii
        return np.ones(len(c_pred)) / len(c_pred), radii

    def step(
        self,
        c_pred: np.ndarray,
        c_true: np.ndarray,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
        A_ub: Optional[np.ndarray] = None,
        b_ub: Optional[np.ndarray] = None,
        bounds: Optional[list] = None,
    ) -> Dict:
        """
        Full online step: get radii, solve LP, observe truth, update.

        This is the main entry point for the online loop.

        Parameters
        ----------
        c_pred : array of shape (d,)
            Predicted cost vector.
        c_true : array of shape (d,)
            True cost vector (observed after decision).
        A_eq, b_eq, A_ub, b_ub, bounds : LP constraints.

        Returns
        -------
        result : dict with keys:
            "z_opt" : LP solution
            "cost" : true cost c^T z*
            "radii" : DICA radii used
            "std_covered" : scalar coverage indicator
            "dica_covered" : DICA-radii coverage indicator
            "poc" : Price of Coverage for this round
        """
        # Solve robust LP with DICA radii
        z_opt, radii = self.solve_robust_lp(
            c_pred, A_eq=A_eq, b_eq=b_eq,
            A_ub=A_ub, b_ub=b_ub, bounds=bounds
        )

        # True cost
        cost = float(c_true @ z_opt)
        self._costs.append(cost)

        # Nominal cost (what you'd pay without robustification)
        res_nom = linprog(
            c_pred, A_eq=A_eq, b_eq=b_eq,
            A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs"
        )
        if res_nom.success:
            nominal_cost = float(c_true @ res_nom.x)
        else:
            nominal_cost = cost
        self._nominal_costs.append(nominal_cost)

        # Update conformal state
        stats = self.update(c_pred, c_true)

        # Feed LP solution back
        self.update_allocation(z_opt)

        # PoC for this round
        poc = (cost / nominal_cost - 1) if nominal_cost > 1e-10 else 0.0

        return {
            "z_opt": z_opt,
            "cost": cost,
            "nominal_cost": nominal_cost,
            "radii": radii,
            "std_covered": stats["std_covered"],
            "dica_covered": stats["dica_covered"],
            "poc": poc,
        }

    def get_results(self) -> Dict[str, float]:
        """Summary statistics after running."""
        avg_poc = float(np.mean(self.poc_history)) if self.poc_history else 0.0
        return {
            "coverage": self.coverage,
            "dica_coverage": self.dica_coverage,
            "avg_poc": avg_poc,
            "avg_cost": float(np.mean(self._costs)) if self._costs else 0.0,
            "n_rounds": len(self._costs),
        }

    def reset(self):
        """Reset all state for a fresh run."""
        self.alpha = self.alpha_target
        self._scores = []
        self._sigma = None
        self._allocation_ema = None
        self._std_coverages = []
        self._dica_coverages = []
        self._costs = []
        self._nominal_costs = []
