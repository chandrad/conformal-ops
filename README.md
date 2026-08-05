# conformal-ops

**Online Conformal Prediction for Predict-then-Optimize**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-54%2F54%20passing-brightgreen.svg)]()

`conformal-ops` brings calibrated, distribution-free uncertainty to **predict-then-optimize**
pipelines (predict a cost vector → solve an LP → observe → repeat). It packages **two published
methods** behind one `scipy.linprog`-style interface, plus five baselines for benchmarking:

| Method | Question it answers | Paper |
|---|---|---|
| **`FreeCoverageDiagnostic`** | *Is coverage free?* — before you hedge, does conformal uncertainty change your decision at all? Returns **free / critical / costly**. | [COPA 2026, PMLR 329](#citation) |
| **`DICA`** | *Make coverage cheaper.* — reshape conformal radii using the optimizer's own allocation, cutting the Price of Coverage **43–54%** while holding **90%** coverage. | [MLHC 2026, PMLR 340](https://proceedings.mlr.press/v340/) |

> **Diagnose, then reduce.** `FreeCoverageDiagnostic` tells you *whether* you have a Price of
> Coverage; `DICA` *reduces* it. Both run on the same `(c_pred, c_true, LP)` inputs.

## Installation

```bash
pip install conformal-ops
```

Or from source:
```bash
git clone https://github.com/chandrad/conformal-ops.git
cd conformal-ops
pip install -e ".[dev]"
```

## Quickstart

Two copy-paste snippets — each runs in ~2 seconds, no data needed.

### A. *Is coverage free?* — `FreeCoverageDiagnostic` (COPA)

Run this **before** deploying conformal hedging: it measures whether coverage changes the decision.

```python
import numpy as np
from conformal_ops import FreeCoverageDiagnostic

# A short calibration pilot: predicted vs true cost vectors over T rounds, and your LP.
rng = np.random.RandomState(0)
d, T = 5, 200
base = np.array([0.1, 1.0, 1.0, 1.0, 1.0])           # item 0 clearly cheapest
c_true = base + rng.normal(0, 0.01, (T, d))
c_pred = c_true + rng.normal(0, 0.02, (T, d))         # an accurate predictor
lp = dict(A_eq=np.ones((1, d)), b_eq=np.array([1.0]), bounds=[(0.0, 1.0)] * d)

report = FreeCoverageDiagnostic(alpha=0.10).run(c_pred, c_true, **lp)
print(report.regime)                  # "free" → coverage never changes the decision
print(report.neutral_frac_committed)  # ~1.00 (funded set unchanged)
# Verdict "costly"? → reduce the premium with DICA (below).
```

### B. *Reduce the Price of Coverage* — `DICA` (MLHC)

When coverage **is** costly, DICA reshapes the radii to cut the premium while holding 90% coverage.

```python
import numpy as np
from conformal_ops import DICA

# 1. Define your LP
d = 20                              # 20 decision variables
A_eq = np.ones((1, d))              # budget constraint: sum(z) = 12
b_eq = np.array([12.0])
bounds = [(0.1, 1.0)] * d           # each variable in [0.1, 1.0]

# 2. Create DICA
dica = DICA(alpha=0.10, beta=0.5)   # 90% coverage target

# 3. Online loop: predict → decide → observe → update
rng = np.random.RandomState(42)
for t in range(300):
    base = rng.exponential(2.0, size=d)
    c_pred = 0.5 + (base + rng.normal(0, 0.5, d)) / 10  # noisy prediction
    c_true = 0.5 + base / 10                              # true cost

    result = dica.step(c_pred, c_true, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    # result["z_opt"]   → LP solution (what to allocate)
    # result["cost"]    → true cost incurred
    # result["poc"]     → Price of Coverage this round

# 4. Check results
print(dica.get_results())
# {'coverage': ~0.90, 'dica_coverage': ~0.90, 'avg_poc': ~0.003, ...}
```

**Same inputs, one interface.** Both methods take `(c_pred, c_true, LP)`; the diagnostic runs a
pilot and returns a verdict, DICA runs the online loop and returns per-round decisions.

### Using with your own predictor and LP

```python
from conformal_ops import DICA

dica = DICA(alpha=0.10, beta=0.5)

for t in range(T):
    c_pred = your_model.predict(features_t)   # your predictor
    # DICA solves the robust LP for you:
    result = dica.step(c_pred, c_true,
                       A_eq=A_eq, b_eq=b_eq,  # your constraints
                       A_ub=A_ub, b_ub=b_ub,  # (optional)
                       bounds=bounds)
    allocation = result["z_opt"]               # use this
```

## How DICA Works

Standard conformal prediction assigns uniform uncertainty margins to every dimension. In an LP, many variables sit at their lower bounds — the margins on these dimensions inflate cost without protecting the decision.

DICA reshapes radii based on LP allocation feedback:

```
r_j^DICA = q_t · σ_j · w_j

where  w_j = ((1 - β) + β · z̄_j / max(z̄)) / w̄
```

- `q_t`: adaptive conformal quantile ([Gibbs & Candès, 2021](#references)) — unchanged
- `σ_j`: per-dimension noise scale (EMA of residuals)
- `w_j`: redistribution weight from allocation EMA
- `β`: redistribution strength (0 = standard, 0.5 = default)

**High allocation → `w_j ≈ 1`** (standard radii preserved)
**Low allocation → `w_j < 1`** (tighter radii, lower cost)

The scalar coverage guarantee (Gibbs-Candès) is preserved — only the radii allocation changes.

## How the diagnostic works

`FreeCoverageDiagnostic` (Quickstart A) runs a short online-conformal calibration pilot on your
predictor and LP, and each round compares the **robust** decision (`argmin (ĉ+r)ᵀx`) against the
**nominal** one (`argmin ĉᵀx`). It reports:

- **`regime`** — `free` (coverage never changes the decision), `critical`, or `costly`.
- **decision-neutrality at two granularities** — the **committed set** (which variables are active)
  and the **full vector** (the exact allocation). These can disagree: a decision can be
  set-stable at 100% while a fractional vertex still shifts, so report the one that matches your
  real decision.
- **`coverage`, `q*`, `σ_q`** — the realized coverage (a calibration sanity check) and the size /
  fluctuation of the conformal quantile over the pilot.
- **analytic κ\* and safety margin** `m = (κ* − q*) / σ_q` — *optional*, when you pass a
  `competitors` oracle (K-shortest-paths for graphs, vertex enumeration for small LPs).

Method: *When Is Conformal Coverage Free? Switching Thresholds for Predict-then-Optimize*
(COPA 2026, PMLR 329).

## Methods

All methods share the same `.step()` interface for easy comparison:

```python
result = method.step(c_pred, c_true, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
# result: {"z_opt", "cost", "poc", "std_covered", "radii", ...}

stats = method.get_results()
# stats: {"coverage", "avg_poc", "avg_cost", "n_rounds"}
```

| Method | Class | Description | Reference |
|--------|-------|-------------|-----------|
| **DICA** | `DICA(beta=0.5)` | Allocation-feedback radii redistribution | [Dronavajjala, 2026](#citation) |
| **UCA** | `UCA()` | Uniform Conformal Allocation (standard online conformal). Equivalent to DICA with β=0. | [Gibbs & Candès, 2021](https://proceedings.neurips.cc/paper/2021/hash/0d441de75e12db29bb25b4e63f23e12f-Abstract.html) |
| **CPO** | `CPO()` | Conformal Predict-then-Optimize. Split conformal with periodic recalibration. Coverage degrades under distribution shift. | [Patel et al., AISTATS 2024](https://proceedings.mlr.press/v238/patel24a.html) |
| **EWMA** | `EWMA()` | Exponential weighted moving average heuristic. No coverage target. Illustrates the gap between ad-hoc heuristics and calibrated methods. | — |
| **ACRO** | `ACRO()` | Group-conditional conformal by patient acuity tercile (Mondrian-style). Tests whether group-level calibration reduces PoC. | Inspired by [Vovk et al., 2003](https://pure.royalholloway.ac.uk/en/publications/mondrian-confidence-machine) |
| **Nominal** | `Nominal()` | Solve LP with predictions directly. No robustification. PoC = 0 by definition. | — |
| **FixedMargin** | `FixedMargin(0.10)` | Add fixed percentage buffer (e.g., 10%). Common operational heuristic. | — |

## What's Inside

```
conformal_ops/
├── core/           # Gibbs-Candès online conformal prediction
├── dica/           # DICA: allocation-feedback radii redistribution
├── baselines/      # UCA, CPO, EWMA, ACRO, Nominal, FixedMargin
├── diagnostics/    # FreeCoverageDiagnostic: is coverage decision-neutral?
└── problems/       # Example LP formulations (nurse/bed/discharge)
```

## How to run

**Install.** The library core is just `numpy` + `scipy`; the demos and notebooks additionally
need `scikit-learn` + `matplotlib` (bundled in the `examples` extra):

```bash
pip install conformal-ops                 # library only (numpy + scipy)
pip install "conformal-ops[examples]"     # + scikit-learn, matplotlib for demos/notebooks
# from source (with tests): pip install -e ".[dev]"
```

**Runnable scripts** (from the repo root, each ~2 s):

```bash
# --- FreeCoverageDiagnostic (COPA) ---
python examples/free_coverage_demo.py     # free / critical / costly verdict

# --- DICA (MLHC) ---
python examples/quickstart.py             # DICA vs UCA: Price-of-Coverage comparison
python examples/nurse_staffing_demo.py    # DICA with 3 plots (needs matplotlib)
python examples/custom_lp.py              # DICA on your own LP
```

**Notebooks** — committed already-executed (they render with outputs on GitHub). Re-run any in
place with:

```bash
jupyter nbconvert --to notebook --execute --inplace examples/<name>.ipynb
```

| Notebook | Method | Data | What it shows |
|---|---|---|---|
| `examples/free_coverage.ipynb` | FreeCoverage (COPA) | synthetic | free vs costly, analytic κ\* |
| `examples/free_coverage_real_data.ipynb` | FreeCoverage (COPA) | California Housing (20K) | committed-set vs vector neutrality, dimension dependence |
| `examples/dica_tutorial.ipynb` | DICA (MLHC) | synthetic | setup, run, visualize, tune β |
| `examples/real_data_healthcare.ipynb` | DICA (MLHC) | UCI Diabetes (100K) | hospital LOS prediction → nurse staffing |
| `examples/real_data_housing.ipynb` | DICA (MLHC) | California Housing (20K) | house value → investment allocation |

**Tests:**

```bash
pip install -e ".[dev]"
pytest -q          # 54 tests, ~7 s
```

## Key Results (from paper)

| Metric | UCA (standard) | DICA | CPO | EWMA |
|--------|---------------|------|-----|------|
| PoC (nurse, MIMIC) | +7.9% | **+4.4%** | +7.3% | +12.4% |
| PoC (discharge, MIMIC) | +13.8% | **+7.5%** | +12.2% | +20.2% |
| Coverage | 90% | 90% | 78–86% | 0–22% |

Validated on 328K patient stays from MIMIC-IV, eICU, and UCI Diabetes.

**DICA reduces PoC by 43–54%** relative to UCA while maintaining the same 90% coverage.

## When to use which

Both methods apply to any online predict-then-optimize setup: a **predictor** emitting
vector-valued cost predictions, an **LP** using them as cost coefficients, and an **online**
setting where true costs are revealed after each decision. They are domain-agnostic — healthcare
staffing, energy dispatch, routing, portfolio/logistics.

- **Start with `FreeCoverageDiagnostic`.** It tells you whether conformal coverage is *free*
  (never changes your decision), *critical*, or *costly* on your problem — a pre-deployment check.
- **If the verdict is `costly`, reach for `DICA`.** It reduces the Price of Coverage, and helps
  most when the LP solution is **sparse** (many variables at their lower bounds, common in
  resource allocation), where uniform conformal radii waste budget on inactive dimensions.

## Coming Soon

- **DIAC**: Dual-Informed Adaptive Conformal (energy/transport networks)
- **Graph conformal**: Conformal prediction on graph-structured optimization

## References

- **Gibbs, I. & Candès, E.** (2021). Adaptive conformal inference under distribution shift. *NeurIPS 34*, 1660–1672. — The foundational online conformal method that DICA builds on.
- **Patel, Y. et al.** (2024). Conformal contextual robust optimization. *AISTATS*, PMLR 238, 1090–1098. — CPO: split conformal for predict-then-optimize (our baseline).
- **Bertsimas, D. & Sim, M.** (2004). The price of robustness. *Operations Research*, 52(1), 35–53. — The "Price of Robustness" concept that inspired our Price of Coverage.
- **Vovk, V. et al.** (2003). Mondrian Confidence Machine. Technical report, Royal Holloway. — Group-conditional conformal prediction (basis for ACRO baseline).
- **Elmachtoub, A. & Grigas, P.** (2022). Smart "Predict, then Optimize". *Management Science*, 68(1), 9–26. — The predict-then-optimize framework.
- **Lei, J. et al.** (2018). Distribution-free predictive inference for regression. *JASA*, 113(523), 1094–1111. — Conformal regression with finite-sample coverage.

## Citation

`DICA` and the package (MLHC 2026, PMLR 340):

```bibtex
@inproceedings{dronavajjala2026dica,
  title={Decision-Informed Online Conformal Prediction for {ICU} Resource Allocation},
  author={Dronavajjala, Chandra Sekhar},
  booktitle={Proceedings of Machine Learning Research},
  volume={340},
  year={2026},
  publisher={PMLR}
}
```

The `FreeCoverageDiagnostic` method (COPA 2026, PMLR 329):

```bibtex
@inproceedings{dronavajjala2026free,
  title={When Is Conformal Coverage Free? Switching Thresholds for Predict-then-Optimize},
  author={Dronavajjala, Chandra Sekhar and Kuppa, Shiva},
  booktitle={Proceedings of Machine Learning Research},
  volume={329},
  year={2026},
  publisher={PMLR},
  note={To appear}
}
```

## License

MIT
