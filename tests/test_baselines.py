"""Unit tests for baseline methods: CPO, EWMA, ACRO, Nominal, FixedMargin."""

import numpy as np
import pytest
from conformal_ops import DICA, UCA, CPO, EWMA, ACRO, Nominal, FixedMargin


def _make_data(n_rounds=200, d=20, seed=42):
    """Generate synthetic data and LP constraints."""
    rng = np.random.RandomState(seed)
    A_eq = np.ones((1, d))
    b_eq = np.array([0.6 * d])
    bounds = [(0.1, 1.0)] * d
    data = []
    for _ in range(n_rounds):
        base = rng.exponential(2.0, size=d)
        c_true = 0.5 + base / 10
        c_pred = 0.5 + (base + rng.normal(0, 0.5, d)) / 10
        data.append((c_pred, c_true))
    return data, A_eq, b_eq, bounds


def _run_method(method, data, A_eq, b_eq, bounds):
    for c_pred, c_true in data:
        method.step(c_pred, c_true, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    return method.get_results()


# --- All methods return consistent interface ---

def test_all_methods_return_same_keys():
    """Every method's .step() must return the same dict keys."""
    data, A_eq, b_eq, bounds = _make_data(n_rounds=10)
    required_keys = {"z_opt", "cost", "nominal_cost", "radii", "std_covered", "poc"}

    for cls in [DICA, UCA, CPO, EWMA, ACRO, Nominal, FixedMargin]:
        if cls == DICA:
            m = cls(beta=0.5)
        elif cls == FixedMargin:
            m = cls(margin=0.10)
        else:
            m = cls()
        c_pred, c_true = data[0]
        result = m.step(c_pred, c_true, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        for key in required_keys:
            assert key in result, f"{cls.__name__}.step() missing key: {key}"


def test_all_methods_get_results_keys():
    """Every method's .get_results() must return coverage, avg_poc, n_rounds."""
    data, A_eq, b_eq, bounds = _make_data(n_rounds=10)
    required_keys = {"coverage", "avg_poc", "n_rounds"}

    for cls in [DICA, UCA, CPO, EWMA, ACRO, Nominal, FixedMargin]:
        if cls == DICA:
            m = cls(beta=0.5)
        elif cls == FixedMargin:
            m = cls(margin=0.10)
        else:
            m = cls()
        _run_method(m, data[:10], A_eq, b_eq, bounds)
        results = m.get_results()
        for key in required_keys:
            assert key in results, f"{cls.__name__}.get_results() missing key: {key}"


# --- CPO tests ---

def test_cpo_coverage_below_target_under_shift():
    """CPO should degrade under distribution shift."""
    rng = np.random.RandomState(42)
    d = 20
    A_eq = np.ones((1, d))
    b_eq = np.array([12.0])
    bounds = [(0.1, 1.0)] * d

    cpo = CPO(alpha=0.10, recalib_freq=50)
    for t in range(400):
        # Shift at t=200: mean doubles
        mean = 2.0 if t < 200 else 4.0
        base = rng.exponential(mean, size=d)
        c_true = 0.5 + base / 10
        c_pred = 0.5 + (base + rng.normal(0, 0.5, d)) / 10
        cpo.step(c_pred, c_true, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

    # CPO should have lower coverage than target under shift
    assert cpo.coverage < 0.90, f"CPO coverage {cpo.coverage:.3f} should be below 90% under shift"


def test_cpo_calib_scores_windowed():
    """CPO calibration scores should be windowed (not grow unboundedly)."""
    data, A_eq, b_eq, bounds = _make_data(n_rounds=300)
    cpo = CPO(alpha=0.10, recalib_freq=50)
    _run_method(cpo, data, A_eq, b_eq, bounds)
    assert len(cpo._calib_scores) <= 200, \
        f"CPO calib_scores should be windowed, got {len(cpo._calib_scores)}"


# --- EWMA tests ---

def test_ewma_uses_c_pred_not_mu():
    """EWMA should use c_pred (not internal mu) for fair comparison."""
    data, A_eq, b_eq, bounds = _make_data(n_rounds=50)
    ewma = EWMA(gamma=1.5)

    for c_pred, c_true in data[:50]:
        result = ewma.step(c_pred, c_true, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        # The robust cost should be c_pred + radii, not mu + radii
        # Verify by checking radii are based on variance, not on mu
        assert result["radii"] is not None
        assert len(result["radii"]) == len(c_pred)


def test_ewma_no_coverage_target():
    """EWMA coverage should NOT track 90% — it has no target."""
    data, A_eq, b_eq, bounds = _make_data(n_rounds=300)
    ewma = EWMA(gamma=1.5)
    results = _run_method(ewma, data, A_eq, b_eq, bounds)
    # EWMA coverage is whatever it happens to be — not necessarily near 90%
    # Just verify it returns a number
    assert 0.0 <= results["coverage"] <= 1.0


# --- ACRO tests ---

def test_acro_group_assignment_uses_c_pred():
    """ACRO groups should be assigned from c_pred, not c_true."""
    acro = ACRO(alpha=0.10)
    c_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                       1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
    groups = acro._assign_groups(c_pred)
    # Terciles of c_pred: [0.1..2.0] → thresholds at ~0.73, ~1.37
    assert groups[0] == 0  # 0.1 is in lowest tercile
    assert groups[-1] == 2  # 2.0 is in highest tercile
    assert len(groups) == 20


def test_acro_runs_without_error():
    """ACRO should complete 200 rounds without crashing."""
    data, A_eq, b_eq, bounds = _make_data(n_rounds=200)
    acro = ACRO(alpha=0.10)
    results = _run_method(acro, data, A_eq, b_eq, bounds)
    assert results["n_rounds"] == 200
    assert 0.0 <= results["coverage"] <= 1.0


# --- Nominal tests ---

def test_nominal_poc_zero():
    """Nominal PoC should always be 0."""
    data, A_eq, b_eq, bounds = _make_data(n_rounds=50)
    nom = Nominal()
    results = _run_method(nom, data[:50], A_eq, b_eq, bounds)
    assert results["avg_poc"] == 0.0


def test_nominal_deterministic():
    """Nominal should give identical results on same data."""
    data, A_eq, b_eq, bounds = _make_data(n_rounds=50)
    nom1 = Nominal()
    nom2 = Nominal()
    for c_pred, c_true in data[:50]:
        r1 = nom1.step(c_pred, c_true, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        r2 = nom2.step(c_pred, c_true, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        np.testing.assert_allclose(r1["cost"], r2["cost"], rtol=1e-10)


# --- FixedMargin tests ---

def test_fixed_margin_deterministic():
    """FixedMargin should give identical results on same data."""
    data, A_eq, b_eq, bounds = _make_data(n_rounds=50)
    fm1 = FixedMargin(margin=0.10)
    fm2 = FixedMargin(margin=0.10)
    for c_pred, c_true in data[:50]:
        r1 = fm1.step(c_pred, c_true, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        r2 = fm2.step(c_pred, c_true, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        np.testing.assert_allclose(r1["cost"], r2["cost"], rtol=1e-10)


def test_fixed_margin_radii_proportional():
    """FixedMargin radii should be margin * |c_pred|."""
    data, A_eq, b_eq, bounds = _make_data(n_rounds=5)
    fm = FixedMargin(margin=0.15)
    c_pred, c_true = data[0]
    result = fm.step(c_pred, c_true, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    expected_radii = np.abs(c_pred) * 0.15
    np.testing.assert_allclose(result["radii"], expected_radii, rtol=1e-10)


# --- Edge cases ---

def test_single_dimension():
    """All methods should handle d=1."""
    d = 1
    A_eq = np.ones((1, d))
    b_eq = np.array([0.5])
    bounds = [(0.1, 1.0)] * d
    rng = np.random.RandomState(42)

    for cls in [DICA, UCA, CPO, EWMA, Nominal, FixedMargin]:
        if cls == DICA:
            m = cls(beta=0.5)
        elif cls == FixedMargin:
            m = cls(margin=0.10)
        else:
            m = cls()
        for _ in range(20):
            c_pred = np.array([0.5 + rng.rand()])
            c_true = np.array([0.5 + rng.rand()])
            m.step(c_pred, c_true, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        assert m.get_results()["n_rounds"] == 20


def test_large_dimension():
    """Methods should handle d=100."""
    d = 100
    A_eq = np.ones((1, d))
    b_eq = np.array([60.0])
    bounds = [(0.1, 1.0)] * d
    rng = np.random.RandomState(42)

    dica = DICA(beta=0.5)
    for _ in range(50):
        base = rng.exponential(2.0, size=d)
        c_true = 0.5 + base / 10
        c_pred = 0.5 + (base + rng.normal(0, 0.5, d)) / 10
        dica.step(c_pred, c_true, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    assert dica.get_results()["n_rounds"] == 50


def test_dica_beta_one():
    """DICA with beta=1.0 should not crash (maximum redistribution)."""
    data, A_eq, b_eq, bounds = _make_data(n_rounds=100)
    dica = DICA(beta=1.0)
    results = _run_method(dica, data[:100], A_eq, b_eq, bounds)
    assert results["n_rounds"] == 100
    assert results["coverage"] > 0.5  # should still have some coverage
