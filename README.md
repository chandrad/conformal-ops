# conformal-ops

**Online Conformal Prediction for Optimization**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-22%2F22%20passing-brightgreen.svg)]()

`conformal-ops` provides methods for integrating online conformal prediction with downstream optimization. It includes **DICA** (Decision-Informed Conformal Adaptation) and five baseline methods — all sharing the same interface for easy benchmarking.

DICA uses LP allocation feedback to reshape conformal radii, reducing the cost of calibrated uncertainty by 43–54% while maintaining 90% coverage.

**Paper:** [Decision-Informed Online Conformal Prediction for ICU Resource Allocation](https://proceedings.mlr.press/v340/) (MLHC 2026, PMLR 340)

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

## 30-Second Quickstart

Copy-paste this. It runs in 2 seconds, no data needed:

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

**That's it.** Three lines to set up, one line per round. DICA handles conformal prediction, radii reshaping, and LP solving internally.

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
└── problems/       # Example LP formulations (nurse/bed/discharge)
```

## Examples & Tutorials

```bash
# Quickstart — DICA vs UCA in 30 lines (< 2 seconds)
python examples/quickstart.py

# Full demo with 3 plots (coverage, PoC, radii)
pip install matplotlib
python examples/nurse_staffing_demo.py

# Use DICA with your own LP
python examples/custom_lp.py
```

**Interactive notebooks:**

| Notebook | Data | Description |
|----------|------|-------------|
| `examples/dica_tutorial.ipynb` | Synthetic | Step-by-step tutorial: setup, run, visualize, tune β |
| `examples/real_data_healthcare.ipynb` | UCI Diabetes (100K) | Real hospital LOS prediction → nurse staffing |
| `examples/real_data_housing.ipynb` | California Housing (20K) | Non-healthcare: house value prediction → investment allocation |

## Key Results (from paper)

| Metric | UCA (standard) | DICA | CPO | EWMA |
|--------|---------------|------|-----|------|
| PoC (nurse, MIMIC) | +7.9% | **+4.4%** | +7.3% | +12.4% |
| PoC (discharge, MIMIC) | +13.8% | **+7.5%** | +12.2% | +20.2% |
| Coverage | 90% | 90% | 78–86% | 0–22% |

Validated on 328K patient stays from MIMIC-IV, eICU, and UCI Diabetes.

**DICA reduces PoC by 43–54%** relative to UCA while maintaining the same 90% coverage.

## When to Use DICA

DICA helps when you have:
1. **A predictor** producing vector-valued cost predictions
2. **An LP** that uses those predictions as cost coefficients
3. **An online setting** where you observe true costs after each decision
4. **LP sparsity** — many variables at their lower bounds (common in resource allocation)

DICA is domain-agnostic: it works for healthcare staffing, energy allocation, portfolio optimization, logistics — any LP with cost uncertainty.

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
# 22 tests, ~5 seconds
```

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

## License

MIT
