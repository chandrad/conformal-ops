"""
Free-Coverage Diagnostic: is conformal coverage decision-neutral?

Before deploying conformal hedging on a predict-then-optimize pipeline, this
tool answers a logically prior question to "how do I reduce the Price of
Coverage": *does coverage change the decision at all?*  If it does not, the
calibrated uncertainty set is a free certificate --- 90% coverage tracking at no
cost to the decision.

The diagnostic runs a short calibration pilot with the deployed predictor and
the same linear-program interface used by the rest of the package, and reports:

  - the MEASURED decision-neutral fraction, at two granularities:
      * committed-set neutrality: does the SET of active variables (support) of
        the robust decision equal that of the nominal decision?
      * vector neutrality: is the full decision vector identical?
    These differ: a decision can be committed-set-stable at 100% while the
    fractional vertex still shifts (e.g. a marginal generator's output).
  - the calibrated quantile process (q*, sigma_q) from the pilot;
  - an optional analytic switching threshold kappa* and safety margin
    m = (kappa* - q*) / sigma_q, when a problem-specific competitor oracle is
    supplied (K-shortest-paths for graphs, vertex enumeration for small LPs);
  - a free / critical / costly regime verdict.

Method reference: "When Is Conformal Coverage Free? Switching Thresholds for
Predict-then-Optimize" (COPA 2026).  The kappa* / safety-margin analysis and the
free/critical/costly triage are from that paper; the online calibration reuses
`OnlineConformal` (Gibbs-Candes, 2021).
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from scipy.optimize import linprog

from conformal_ops.core.online_conformal import OnlineConformal

ArrayLike = Union[np.ndarray, Sequence[np.ndarray]]


@dataclass
class FreeCoverageReport:
    """Result of a :class:`FreeCoverageDiagnostic` run.

    Attributes
    ----------
    regime : str
        One of ``"free"``, ``"critical"``, ``"costly"``.
    neutral_frac_committed : float
        Fraction of (post-warmup) rounds on which the robust and nominal
        decisions have the SAME support (set of active variables).
    neutral_frac_vector : float
        Fraction of rounds on which the full robust and nominal decision
        vectors are identical.  Always ``<= neutral_frac_committed``.
    coverage : float
        Empirical joint coverage of the conformal sets over the pilot
        (averaged over ALL rounds including warmup; early rounds are trivially
        covered while the score buffer fills, so this can be mildly optimistic).
    q_star : float
        Mean conformal quantile over the (post-warmup) pilot.
    sigma_q : float
        Std of the conformal quantile over the (post-warmup) pilot --- the
        empirical width of the quantile process.
    n_rounds : int
        Number of pilot rounds that produced a valid (feasible) decision.
    warmup : int
        Number of leading rounds excluded from the aggregates.
    kappa_star : float or None
        Analytic switching threshold, if a competitor oracle was supplied.
    margin : float or None
        Safety margin ``(kappa_star - q_star) / sigma_q``, if available.
    verdict_basis : str
        ``"margin"`` if the regime came from the analytic margin, else
        ``"empirical"`` (from the measured committed-set neutral fraction).
    """

    regime: str
    neutral_frac_committed: float
    neutral_frac_vector: float
    coverage: float
    q_star: float
    sigma_q: float
    n_rounds: int
    warmup: int
    kappa_star: Optional[float] = None
    margin: Optional[float] = None
    verdict_basis: str = "empirical"

    def __repr__(self) -> str:
        lines = [
            f"FreeCoverageReport(regime={self.regime!r}, basis={self.verdict_basis})",
            f"  decision-neutral (committed set): {self.neutral_frac_committed:6.1%}",
            f"  decision-neutral (full vector):   {self.neutral_frac_vector:6.1%}",
            f"  coverage:                         {self.coverage:6.1%}",
            f"  q* = {self.q_star:.4f}   sigma_q = {self.sigma_q:.4f}",
        ]
        if self.kappa_star is not None:
            km = "inf" if np.isinf(self.kappa_star) else f"{self.kappa_star:.4f}"
            mm = "inf" if (self.margin is not None and np.isinf(self.margin)) else \
                (f"{self.margin:.2f}" if self.margin is not None else "n/a")
            lines.append(f"  kappa* = {km}   margin m = {mm}")
        lines.append(f"  rounds = {self.n_rounds} (warmup {self.warmup})")
        return "\n".join(lines)


class FreeCoverageDiagnostic:
    """Diagnose whether conformal coverage is decision-neutral for an LP.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate (0.10 -> 90% coverage).
    eta, window, scale_ema : float, int, float
        Online-conformal (Gibbs-Candes) hyperparameters; passed through to
        :class:`OnlineConformal`.
    warmup : int
        Leading pilot rounds excluded from the aggregates (the conformal layer
        uses a conservative default radius until the score buffer fills).
    free_margin, costly_margin : float
        Safety-margin cutoffs for the analytic verdict: ``m > free_margin`` ->
        free, ``m < costly_margin`` -> costly, otherwise critical.
    free_neutral, costly_neutral : float
        Committed-set neutral-fraction cutoffs for the empirical verdict when no
        competitor oracle is supplied.
    support_eps : float
        Relative threshold for deciding a variable is "active" (in the support):
        ``x_j`` is active iff ``x_j > support_eps * max(max|x|, 1)``.
    vector_atol : float
        Absolute tolerance for full-vector equality (``np.allclose``).
    """

    def __init__(
        self,
        alpha: float = 0.10,
        eta: float = 0.05,
        window: int = 150,
        scale_ema: float = 0.05,
        warmup: int = 20,
        free_margin: float = 2.0,
        costly_margin: float = -2.0,
        free_neutral: float = 0.95,
        costly_neutral: float = 0.50,
        support_eps: float = 1e-3,
        vector_atol: float = 1e-6,
    ):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if free_neutral <= costly_neutral:
            raise ValueError("free_neutral must exceed costly_neutral")
        if costly_margin >= free_margin:
            raise ValueError("costly_margin must be below free_margin")
        self.alpha = alpha
        self.eta = eta
        self.window = window
        self.scale_ema = scale_ema
        self.warmup = warmup
        self.free_margin = free_margin
        self.costly_margin = costly_margin
        self.free_neutral = free_neutral
        self.costly_neutral = costly_neutral
        self.support_eps = support_eps
        self.vector_atol = vector_atol

    # -- LP solve (same interface as DICA / baselines) ---------------------
    @staticmethod
    def _solve(c, cons) -> Optional[np.ndarray]:
        res = linprog(c, method="highs", **cons)
        return res.x if res.success else None

    def _support(self, x: np.ndarray) -> np.ndarray:
        thr = self.support_eps * max(float(np.max(np.abs(x))), 1.0)
        return x > thr

    @staticmethod
    def _as_2d(stream: ArrayLike) -> np.ndarray:
        arr = np.asarray(stream, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError("stream must be shape (T, d) or a length-T list of (d,) arrays")
        return arr

    def run(
        self,
        c_pred_stream: ArrayLike,
        c_true_stream: ArrayLike,
        *,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
        A_ub: Optional[np.ndarray] = None,
        b_ub: Optional[np.ndarray] = None,
        bounds: Optional[list] = None,
        competitors: Optional[Callable[[np.ndarray, np.ndarray], List[np.ndarray]]] = None,
    ) -> FreeCoverageReport:
        """Run the calibration pilot and return a :class:`FreeCoverageReport`.

        Parameters
        ----------
        c_pred_stream, c_true_stream : array (T, d) or list of (d,) arrays
            Predicted and true cost vectors for the T pilot rounds.
        A_eq, b_eq, A_ub, b_ub, bounds : LP constraints (as in ``scipy.linprog``).
        competitors : callable, optional
            ``competitors(c_pred_rep, x_nom_rep) -> list of vertex arrays``: the
            competing vertices against which kappa* is measured (e.g. the
            K-shortest paths, or enumerated LP vertices).  If supplied, the
            report includes an analytic kappa* and safety margin.
        """
        C_pred = self._as_2d(c_pred_stream)
        C_true = self._as_2d(c_true_stream)
        if C_pred.shape != C_true.shape:
            raise ValueError("c_pred_stream and c_true_stream must have the same shape")
        T = C_pred.shape[0]
        if T <= self.warmup:
            raise ValueError(f"pilot length {T} must exceed warmup {self.warmup}")

        cons = dict(A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        oc = OnlineConformal(
            alpha=self.alpha, eta=self.eta, window=self.window, scale_ema=self.scale_ema
        )

        committed: List[float] = []
        vector: List[float] = []
        q_trace: List[float] = []

        for t in range(T):
            c_pred = C_pred[t]
            c_true = C_true[t]
            # Radii from PAST scores only (OnlineConformal state is pre-update).
            radii = oc.get_radii(c_pred)
            x_nom = self._solve(c_pred, cons)
            x_rob = self._solve(c_pred + radii, cons)

            if x_nom is not None and x_rob is not None and t >= self.warmup:
                q_trace.append(oc.quantile)  # current q_t, pre-update
                vec = bool(np.allclose(x_nom, x_rob, rtol=1e-4, atol=self.vector_atol))
                # Committed-set neutrality is strictly weaker than vector
                # neutrality: identical vectors are trivially support-identical.
                # Force the implication so the invariant
                # neutral_frac_committed >= neutral_frac_vector always holds,
                # even when a component sits within vector_atol of the support
                # threshold.
                comm = vec or bool(
                    np.array_equal(self._support(x_nom), self._support(x_rob))
                )
                vector.append(1.0 if vec else 0.0)
                committed.append(1.0 if comm else 0.0)

            # Advance conformal state AFTER the decisions (no look-ahead).
            oc.update(c_pred, c_true)

        n = len(committed)
        if n == 0:
            raise RuntimeError("no feasible post-warmup rounds; check LP constraints")

        neutral_committed = float(np.mean(committed))
        neutral_vector = float(np.mean(vector))
        q_star = float(np.mean(q_trace))
        sigma_q = float(np.std(q_trace))
        coverage = oc.coverage

        kappa_star: Optional[float] = None
        margin: Optional[float] = None
        basis = "empirical"

        if competitors is not None:
            kappa_star = self._kappa_star(C_pred, cons, oc, competitors)
            if kappa_star is not None and sigma_q > 1e-12:
                margin = (kappa_star - q_star) / sigma_q
                basis = "margin"

        if basis == "margin":
            regime = self._regime_from_margin(margin)
        else:
            regime = self._regime_from_neutral(neutral_committed)

        return FreeCoverageReport(
            regime=regime,
            neutral_frac_committed=neutral_committed,
            neutral_frac_vector=neutral_vector,
            coverage=coverage,
            q_star=q_star,
            sigma_q=sigma_q,
            n_rounds=n,
            warmup=self.warmup,
            kappa_star=kappa_star,
            margin=margin,
            verdict_basis=basis,
        )

    # -- analytic kappa* ---------------------------------------------------
    def _kappa_star(self, C_pred, cons, oc, competitors) -> Optional[float]:
        """kappa* = min_{k: delta_k>0} Delta_k / delta_k at a representative cost.

        Delta_k = c^T(v_k - v*)  (cost gap, >= 0 since v* is optimal)
        delta_k = sigma^T(v* - v_k)  (noise-weighted capacity difference)
        """
        sigma = oc._sigma
        if sigma is None:
            return None
        c_rep = C_pred.mean(axis=0)
        v_star = self._solve(c_rep, cons)
        if v_star is None:
            return None
        vertices = competitors(c_rep, v_star)
        best = np.inf
        for v_k in vertices:
            v_k = np.asarray(v_k, dtype=float)
            if v_k.shape != v_star.shape:
                continue
            delta_k = float(sigma @ (v_star - v_k))
            if delta_k > 1e-12:
                delta_cost = float(c_rep @ (v_k - v_star))  # >= 0 (v* optimal)
                if delta_cost < 0.0:
                    # Oracle returned a vertex cheaper than the nominal optimum
                    # (inconsistent oracle or numerical noise): a negative ratio
                    # would spuriously drive kappa* negative. Skip it.
                    continue
                ratio = delta_cost / delta_k
                if ratio < best:
                    best = ratio
        return best  # np.inf if no positive-delta competitor (coverage always free)

    def _regime_from_margin(self, m: float) -> str:
        if m > self.free_margin:
            return "free"
        if m < self.costly_margin:
            return "costly"
        return "critical"

    def _regime_from_neutral(self, frac: float) -> str:
        if frac >= self.free_neutral:
            return "free"
        if frac <= self.costly_neutral:
            return "costly"
        return "critical"
