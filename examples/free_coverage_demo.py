"""
Free-Coverage Diagnostic demo.

Before you spend effort *reducing* the Price of Coverage (e.g. with DICA), ask a
prior question: does conformal coverage change your decision at all? If it does
not, the calibrated uncertainty set is a FREE certificate --- 90% coverage
tracking at zero cost to the decision.

`FreeCoverageDiagnostic` runs a short calibration pilot with your predictor and
your LP, and returns a free / critical / costly verdict plus the measured
decision-neutral fraction at two granularities (committed set vs full vector).

Run:  python examples/free_coverage_demo.py
"""

import numpy as np

from conformal_ops import FreeCoverageDiagnostic


def selection_lp(d):
    """min c^T x  s.t.  sum(x) = 1, 0 <= x <= 1  (optimum is a vertex e_k)."""
    return dict(A_eq=np.ones((1, d)), b_eq=np.array([1.0]), bounds=[(0.0, 1.0)] * d)


def make_stream(T, base, true_noise, pred_noise, seed):
    """pred_noise may be a scalar or a per-component vector. Heterogeneous
    per-component noise is what makes conformal radii reorder the argmin."""
    rng = np.random.RandomState(seed)
    d = len(base)
    pred_noise = np.broadcast_to(pred_noise, (d,))
    c_pred = np.zeros((T, d))
    c_true = np.zeros((T, d))
    for t in range(T):
        ct = base + rng.normal(0, true_noise, d)
        c_true[t] = ct
        c_pred[t] = ct + rng.normal(0, 1.0, d) * pred_noise
    return c_pred, c_true


def main():
    print("=" * 68)
    print("FreeCoverageDiagnostic: is conformal coverage free for this problem?")
    print("=" * 68)

    # ---- Case 1: wide cost gap + accurate predictor -> coverage is FREE ----
    d = 5
    c_pred, c_true = make_stream(
        T=200,
        base=np.array([0.1, 1.0, 1.0, 1.0, 1.0]),  # item 0 clearly cheapest
        true_noise=0.01,
        pred_noise=0.02,
        seed=0,
    )
    free = FreeCoverageDiagnostic(alpha=0.10).run(c_pred, c_true, **selection_lp(d))
    print("\n[Case 1] wide gap, accurate predictor")
    print(free)

    # ---- Case 2: near-tied costs + HETEROGENEOUS noise -> coverage is COSTLY
    # Item 1 is (often) the nominal cheapest but by far the noisiest, so its
    # conformal radius is large and the robust optimum keeps switching away.
    c_pred, c_true = make_stream(
        T=200,
        base=np.array([0.50, 0.48, 0.51, 0.50, 0.505]),  # near-tied, item 1 low
        true_noise=0.01,
        pred_noise=np.array([0.03, 0.60, 0.03, 0.03, 0.03]),  # item 1 very noisy
        seed=1,
    )
    costly = FreeCoverageDiagnostic(alpha=0.10).run(c_pred, c_true, **selection_lp(d))
    print("\n[Case 2] tiny gaps, heterogeneous predictor noise")
    print(costly)

    # ---- Optional: analytic kappa* + safety margin via a competitor oracle -
    # For the selection LP the competing vertices are the other unit vectors.
    c_pred, c_true = make_stream(
        T=200,
        base=np.array([0.1, 1.0, 1.0, 1.0, 1.0]),
        true_noise=0.01,
        pred_noise=0.02,
        seed=0,
    )

    def competitors(c_rep, x_nom):
        star = int(np.argmax(x_nom))
        return [np.eye(d)[k] for k in range(d) if k != star]

    analytic = FreeCoverageDiagnostic(alpha=0.10).run(
        c_pred, c_true, competitors=competitors, **selection_lp(d)
    )
    print("\n[Case 3] same as Case 1, with analytic kappa* / safety margin")
    print(analytic)

    print("\nTakeaway: run this BEFORE deploying conformal hedging. If the")
    print("verdict is 'free', coverage is a zero-cost certificate; if 'costly',")
    print("budget the premium (or reduce it with DICA).")


if __name__ == "__main__":
    main()
