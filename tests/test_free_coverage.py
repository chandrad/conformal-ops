"""Tests for FreeCoverageDiagnostic (COPA 2026 free-coverage diagnostic)."""

import numpy as np
import pytest

from conformal_ops import FreeCoverageDiagnostic, FreeCoverageReport


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _simplex(d):
    """Selection LP: min c^T x s.t. sum(x)=1, 0<=x<=1 (optimum is a vertex e_k)."""
    return dict(A_eq=np.ones((1, d)), b_eq=np.array([1.0]), bounds=[(0.0, 1.0)] * d)


def _stream(T, d, base, true_noise, pred_noise, seed):
    """pred_noise may be scalar or a per-component (d,) vector."""
    rng = np.random.RandomState(seed)
    pred_noise = np.broadcast_to(pred_noise, (d,))
    Cp = np.zeros((T, d))
    Ct = np.zeros((T, d))
    for t in range(T):
        c_true = base + rng.normal(0, true_noise, d)
        c_pred = c_true + rng.normal(0, 1.0, d) * pred_noise
        Ct[t] = c_true
        Cp[t] = c_pred
    return Cp, Ct


# --------------------------------------------------------------------------
# regime behaviour
# --------------------------------------------------------------------------
def test_free_regime_wide_gap_accurate_predictor():
    """Large cost gap + accurate predictor -> coverage is decision-neutral."""
    d = 5
    base = np.array([0.1, 1.0, 1.0, 1.0, 1.0])  # item 0 clearly cheapest
    Cp, Ct = _stream(120, d, base, true_noise=0.01, pred_noise=0.02, seed=0)
    rep = FreeCoverageDiagnostic(alpha=0.10).run(Cp, Ct, **_simplex(d))
    assert isinstance(rep, FreeCoverageReport)
    assert rep.neutral_frac_committed >= 0.95
    assert rep.regime == "free"


def test_costly_regime_tiny_gap_heterogeneous_noise():
    """Tiny cost gaps + HETEROGENEOUS noise -> radii reorder the argmin often.

    Item 1 is (often) the nominal cheapest but by far the noisiest, so its large
    conformal radius makes the robust optimum switch away on many rounds.
    Homogeneous noise would NOT do this (uniform radii preserve the argmin) ---
    the heterogeneity is the point.
    """
    d = 5
    base = np.array([0.50, 0.48, 0.51, 0.50, 0.505])
    noise = np.array([0.03, 0.60, 0.03, 0.03, 0.03])  # item 1 very noisy
    Cp, Ct = _stream(200, d, base, true_noise=0.01, pred_noise=noise, seed=1)
    rep = FreeCoverageDiagnostic(alpha=0.10).run(Cp, Ct, **_simplex(d))
    assert rep.neutral_frac_committed <= 0.50
    assert rep.regime == "costly"


# --------------------------------------------------------------------------
# core invariant: committed-set neutrality is weaker than vector neutrality
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(6))
def test_committed_ge_vector_invariant(seed):
    d = 6
    base = np.array([0.2, 0.5, 0.5, 0.8, 0.5, 0.5])
    Cp, Ct = _stream(140, d, base, true_noise=0.05, pred_noise=0.15, seed=seed)
    rep = FreeCoverageDiagnostic(alpha=0.10).run(Cp, Ct, **_simplex(d))
    assert rep.neutral_frac_committed >= rep.neutral_frac_vector - 1e-12


def test_committed_ge_vector_near_threshold_stress():
    """Components engineered to sit near the support threshold still respect
    committed >= vector (the `comm = vec or ...` guard)."""
    d = 4
    diag = FreeCoverageDiagnostic(alpha=0.10, warmup=5, support_eps=1e-3)
    rng = np.random.RandomState(7)
    T = 60
    Cp = np.zeros((T, d))
    Ct = np.zeros((T, d))
    base = np.array([0.3, 0.3001, 0.7, 0.7])  # first two nearly tied
    for t in range(T):
        Ct[t] = base + rng.normal(0, 1e-3, d)
        Cp[t] = Ct[t] + rng.normal(0, 5e-3, d)
    rep = diag.run(Cp, Ct, **_simplex(d))
    assert rep.neutral_frac_committed >= rep.neutral_frac_vector - 1e-12


# --------------------------------------------------------------------------
# analytic kappa* path
# --------------------------------------------------------------------------
def test_analytic_kappa_star_with_oracle():
    d = 5
    base = np.array([0.1, 1.0, 1.0, 1.0, 1.0])
    Cp, Ct = _stream(120, d, base, true_noise=0.01, pred_noise=0.02, seed=3)

    def competitors(c_rep, x_nom):
        # vertices of the selection simplex other than the nominal optimum
        star = int(np.argmax(x_nom))
        return [np.eye(d)[k] for k in range(d) if k != star]

    rep = FreeCoverageDiagnostic(alpha=0.10).run(
        Cp, Ct, competitors=competitors, **_simplex(d)
    )
    assert rep.kappa_star is not None
    assert rep.margin is not None
    assert rep.verdict_basis == "margin"
    # wide gap + accurate predictor -> comfortably free
    assert rep.regime == "free"


def test_kappa_star_infinite_when_no_positive_delta():
    """If every competitor has equal noise scale, K+ is empty -> kappa*=inf."""
    d = 4
    base = np.array([0.1, 0.9, 0.9, 0.9])
    Cp, Ct = _stream(90, d, base, true_noise=0.02, pred_noise=0.02, seed=4)

    def competitors(c_rep, x_nom):
        # return the nominal optimum itself -> delta_k == 0, skipped -> kappa*=inf
        return [x_nom.copy()]

    rep = FreeCoverageDiagnostic(alpha=0.10).run(
        Cp, Ct, competitors=competitors, **_simplex(d)
    )
    assert np.isinf(rep.kappa_star)
    assert rep.regime == "free"


# --------------------------------------------------------------------------
# determinism & report
# --------------------------------------------------------------------------
def test_deterministic():
    d = 5
    base = np.array([0.1, 1.0, 1.0, 1.0, 1.0])
    Cp, Ct = _stream(100, d, base, true_noise=0.01, pred_noise=0.05, seed=5)
    r1 = FreeCoverageDiagnostic(alpha=0.10).run(Cp, Ct, **_simplex(d))
    r2 = FreeCoverageDiagnostic(alpha=0.10).run(Cp, Ct, **_simplex(d))
    assert r1.neutral_frac_committed == r2.neutral_frac_committed
    assert r1.q_star == r2.q_star
    assert r1.coverage == r2.coverage


def test_repr_contains_regime():
    d = 5
    base = np.array([0.1, 1.0, 1.0, 1.0, 1.0])
    Cp, Ct = _stream(80, d, base, true_noise=0.01, pred_noise=0.02, seed=6)
    rep = FreeCoverageDiagnostic(alpha=0.10).run(Cp, Ct, **_simplex(d))
    s = repr(rep)
    assert rep.regime in s
    assert "decision-neutral" in s


# --------------------------------------------------------------------------
# input validation / edge cases
# --------------------------------------------------------------------------
def test_warmup_must_be_below_pilot_length():
    d = 3
    Cp, Ct = _stream(15, d, np.ones(d) * 0.5, 0.01, 0.02, seed=0)
    with pytest.raises(ValueError):
        FreeCoverageDiagnostic(warmup=20).run(Cp, Ct, **_simplex(d))


def test_shape_mismatch_raises():
    d = 4
    Cp, _ = _stream(50, d, np.ones(d) * 0.5, 0.01, 0.02, seed=0)
    _, Ct = _stream(50, d + 1, np.ones(d + 1) * 0.5, 0.01, 0.02, seed=1)
    with pytest.raises(ValueError):
        FreeCoverageDiagnostic().run(Cp, Ct, **_simplex(d))


def test_bad_alpha_raises():
    with pytest.raises(ValueError):
        FreeCoverageDiagnostic(alpha=1.5)


def test_bad_cutoffs_raise():
    with pytest.raises(ValueError):
        FreeCoverageDiagnostic(free_neutral=0.3, costly_neutral=0.5)
    with pytest.raises(ValueError):
        FreeCoverageDiagnostic(free_margin=-2.0, costly_margin=2.0)
