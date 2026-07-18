"""
How to use DICA with your own optimization problem.

DICA works with any LP. You just need:
1. A predicted cost vector c_pred
2. A true cost vector c_true (observed after decision)
3. LP constraints (A_eq, b_eq, A_ub, b_ub, bounds)

This example shows a simple portfolio-like allocation problem.
"""

import numpy as np
from conformal_ops import DICA

# Your custom LP: allocate budget across 10 assets
d = 10
A_eq = np.ones((1, d))   # sum(z) = 1 (fully invested)
b_eq = np.array([1.0])
bounds = [(0.0, 0.5)] * d  # max 50% in any single asset

# Create DICA instance
dica = DICA(alpha=0.10, beta=0.5)

# Online loop
rng = np.random.RandomState(123)
for t in range(200):
    # Your predictor produces c_pred; nature reveals c_true
    c_true = rng.exponential(1.0, size=d)
    c_pred = c_true + rng.normal(0, 0.3, size=d)

    # One-line DICA step: get radii, solve robust LP, update
    result = dica.step(
        c_pred, c_true,
        A_eq=A_eq, b_eq=b_eq, bounds=bounds
    )

    # result contains everything you need:
    # result["z_opt"]        — LP solution
    # result["cost"]         — true cost
    # result["poc"]          — Price of Coverage this round
    # result["std_covered"]  — scalar coverage indicator
    # result["dica_covered"] — DICA-radii coverage indicator

# Summary
stats = dica.get_results()
print(f"Coverage: {stats['coverage']:.1%}")
print(f"DICA-radii coverage: {stats['dica_coverage']:.1%}")
print(f"Average PoC: {stats['avg_poc']:+.1%}")
print(f"Rounds: {stats['n_rounds']}")
