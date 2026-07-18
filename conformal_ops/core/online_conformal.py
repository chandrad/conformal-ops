"""
Online Conformal Prediction with Normalized Max-Residual Score.

Implements Gibbs-Candes (2021) adaptive conformal inference for
distribution-free coverage under arbitrary distribution shift.

Joint coverage via normalized max-residual:
    s_t = max_j |e_j| / sigma_j       (single scalar score)
    r_j = q_t * sigma_j               (heterogeneous radii)
    P(all j: |e_j| <= r_j) = P(s_t <= q_t)  (exact identity)
"""

import numpy as np
from typing import Optional, List, Dict


class OnlineConformal:
    """
    Standard online conformal prediction (Gibbs-Candes, 2021).

    Maintains a scalar quantile q_t that adapts online to track a target
    coverage rate. Produces per-dimension radii r_j = q_t * sigma_j.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate (default 0.10 for 90% coverage).
    eta : float
        Step size for the Gibbs-Candes alpha update.
    window : int
        Number of recent scores for quantile estimation.
    scale_ema : float
        EMA decay for per-component noise scale sigma_j.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        eta: float = 0.05,
        window: int = 150,
        scale_ema: float = 0.05,
    ):
        self.alpha_target = alpha
        self.alpha = alpha
        self.eta = eta
        self.window = window
        self.scale_ema = scale_ema

        self._scores: List[float] = []
        self._sigma: Optional[np.ndarray] = None
        self._coverages: List[float] = []

    @property
    def coverage(self) -> float:
        """Empirical coverage so far."""
        return float(np.mean(self._coverages)) if self._coverages else 0.0

    @property
    def quantile(self) -> float:
        """Current conformal quantile q_t."""
        return self._get_q()

    def _get_q(self) -> float:
        if len(self._scores) < 5:
            return 1.0
        level = min(1 - self.alpha, 1.0)
        return float(np.quantile(self._scores, level))

    def get_radii(self, c_pred: np.ndarray) -> np.ndarray:
        """
        Compute conformal radii r_j = q_t * sigma_j.

        Parameters
        ----------
        c_pred : array of shape (d,)
            Predicted cost vector.

        Returns
        -------
        radii : array of shape (d,)
            Per-dimension conformal radii.
        """
        d = len(c_pred)
        q = self._get_q()
        if self._sigma is None:
            return np.abs(c_pred) * 0.3
        return np.maximum(q * self._sigma[:d], 0.0)

    def update(self, c_pred: np.ndarray, c_true: np.ndarray) -> float:
        """
        Observe true costs, compute score, update quantile.

        Timing (critical): score uses pre-update sigma, then sigma is updated.

        Parameters
        ----------
        c_pred : array of shape (d,)
            Predicted cost vector.
        c_true : array of shape (d,)
            True (observed) cost vector.

        Returns
        -------
        covered : float
            1.0 if covered (s_t <= q_t), 0.0 otherwise.
        """
        residuals = np.abs(c_true - c_pred)

        if self._sigma is None:
            self._sigma = residuals + 1e-6
            score = 1.0
        else:
            d = len(c_pred)
            if len(self._sigma) != d:
                self._sigma = residuals + 1e-6
                score = 1.0
            else:
                score = float(np.max(residuals / self._sigma))
                self._sigma = (
                    (1 - self.scale_ema) * self._sigma
                    + self.scale_ema * residuals
                )
                self._sigma = np.maximum(self._sigma, 1e-6)

        self._scores.append(score)
        if len(self._scores) > self.window:
            self._scores = self._scores[-self.window:]

        q = self._get_q()
        covered = float(score <= q)
        self._coverages.append(covered)

        err = 1.0 - covered
        self.alpha = float(np.clip(
            self.alpha + self.eta * (self.alpha_target - err), 0.01, 0.5
        ))

        return covered

    def get_stats(self) -> Dict[str, float]:
        """Return current state statistics."""
        return {
            "alpha": self.alpha,
            "quantile": self._get_q(),
            "n_scores": len(self._scores),
            "coverage": self.coverage,
        }

    def reset(self):
        """Reset all state."""
        self.alpha = self.alpha_target
        self._scores = []
        self._sigma = None
        self._coverages = []
