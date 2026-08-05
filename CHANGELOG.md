# Changelog

## 0.2.0

### Added
- **`FreeCoverageDiagnostic`** (`conformal_ops.diagnostics`): a pre-deployment
  diagnostic that answers *is conformal coverage decision-neutral (free)?* for a
  predict-then-optimize LP. Runs a short calibration pilot and returns a
  **free / critical / costly** verdict via `FreeCoverageReport`, reporting
  decision-neutrality at two granularities — **committed set** (support) and
  **full vector** — plus the calibrated quantile process `(q*, sigma_q)` and an
  optional analytic switching threshold `kappa*` / safety margin when a
  problem-specific `competitors` oracle is supplied.
  Method: *When Is Conformal Coverage Free? Switching Thresholds for
  Predict-then-Optimize* (COPA 2026). Complements `DICA`: `DICA` *reduces* the
  Price of Coverage, `FreeCoverageDiagnostic` tells you whether you have one.
- Examples: `examples/free_coverage_demo.py`, `examples/free_coverage.ipynb`
  (synthetic), and `examples/free_coverage_real_data.ipynb` (real California
  Housing — committed-set vs vector neutrality and dimension dependence).
- 17 unit tests for the diagnostic (54 total).

### Notes
- No new **core** dependencies — the diagnostic uses only `numpy` + `scipy`.
  Example-only dependencies (`scikit-learn`, `matplotlib`) live in the `examples`
  / `dev` extras, so `pip install conformal-ops` stays minimal.

## 0.1.0

- Initial release: `DICA`, `OnlineConformal`, and baselines
  (`UCA`, `CPO`, `EWMA`, `ACRO`, `Nominal`, `FixedMargin`).
