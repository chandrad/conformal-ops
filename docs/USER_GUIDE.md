# User Guide

## What is DICA?

**Decision-Informed Conformal Adaptation** reshapes uncertainty margins based on how your optimizer actually uses them.

Standard conformal prediction gives every dimension the same uncertainty buffer. But in an LP, some variables get high allocation (critical patients) while others sit at the minimum (stable patients). The buffer on stable patients is wasted — it inflates cost without protecting the decision.

DICA redistributes: tighter margins on low-allocation dimensions (saving cost), wider on high-allocation dimensions (strengthening protection). The total budget is preserved, so the coverage guarantee on the scalar conformal set is unchanged.

## When should you use DICA?

Use DICA when you have:
1. **A predictor** that produces vector-valued predictions (e.g., LOS for multiple patients)
2. **An LP** that uses those predictions as cost coefficients
3. **An online setting** where predictions arrive sequentially and you observe true values

DICA helps most when the LP solution is **sparse** — many variables at their lower bounds. This is common in resource allocation (staffing, scheduling, inventory).

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

```python
import numpy as np
from conformal_ops import DICA

# Your LP: allocate budget across 20 patients
d = 20
A_eq = np.ones((1, d))
b_eq = np.array([12.0])  # total budget
bounds = [(0.1, 1.0)] * d

# Create DICA
dica = DICA(alpha=0.10, beta=0.5)

# Online loop
for t in range(500):
    c_pred = ...  # your predictor's output
    c_true = ...  # observed after decision

    result = dica.step(c_pred, c_true,
                       A_eq=A_eq, b_eq=b_eq, bounds=bounds)

    z_opt = result["z_opt"]   # LP solution
    cost = result["cost"]     # true cost
    poc = result["poc"]       # Price of Coverage this round

# Summary
print(dica.get_results())
```

## Understanding the Output

`dica.step()` returns a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `z_opt` | array | LP optimal solution under DICA radii |
| `cost` | float | True cost: c_true @ z_opt |
| `nominal_cost` | float | Cost without robustification |
| `radii` | array | DICA radii used this round |
| `std_covered` | float | 1.0 if scalar coverage holds (Gibbs-Candes) |
| `dica_covered` | float | 1.0 if all residuals within DICA radii |
| `poc` | float | Price of Coverage: cost/nominal_cost - 1 |

`dica.get_results()` returns aggregate stats:

| Key | Description |
|-----|-------------|
| `coverage` | Scalar coverage rate (should track alpha_target) |
| `dica_coverage` | Coverage under reshaped radii (empirical) |
| `avg_poc` | Average Price of Coverage |
| `n_rounds` | Number of rounds completed |

## Tuning beta

`beta` controls redistribution strength:
- `beta=0`: Standard conformal (UCA). No redistribution.
- `beta=0.5`: Default. Good balance of cost savings and stability.
- `beta=1`: Fully allocation-proportional. May over-distort on some problems.

The optimal beta is typically 0.3-0.7 (U-shaped PoC curve). Start with 0.5.

## Using with your own LP

DICA works with any LP. You need:

```python
from conformal_ops import DICA

dica = DICA(alpha=0.10, beta=0.5)

# Option 1: Full step (DICA handles the LP solve)
result = dica.step(c_pred, c_true,
                   A_eq=A_eq, b_eq=b_eq,
                   A_ub=A_ub, b_ub=b_ub,
                   bounds=bounds)

# Option 2: Manual control (you solve the LP yourself)
radii = dica.get_radii(c_pred)
c_robust = c_pred + radii
z_opt = your_solver(c_robust, constraints)
dica.update(c_pred, c_true)
dica.update_allocation(z_opt)
```

## FAQ

**Q: Does DICA guarantee 90% coverage?**
A: The Gibbs-Candes guarantee applies to the *scalar* conformal set (s_t <= q_t), which DICA preserves. The *reshaped* DICA radii do not carry a formal per-component guarantee. Empirically, DICA-radii coverage is 88.9-91.1% across our experiments.

**Q: Where do coverage misses happen?**
A: Only on lower-bound dimensions (patients at minimum allocation). High-allocation dimensions are always covered when scalar coverage holds. The clinical cost of lower-bound misses is negligible (< 0.1% of total allocation).

**Q: Can I use DICA with integer programs?**
A: DICA currently works with LP relaxations. For IPs, you'd need to define "allocation importance" for combinatorial solutions — this is an open research direction.

**Q: What if my LP changes structure between rounds?**
A: DICA adapts via EMA, so gradually changing LP structure is fine. If the structure changes abruptly, the allocation EMA needs time to adjust (controlled by `allocation_ema_decay`).
