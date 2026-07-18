"""
DICA Quickstart — Compare DICA vs standard conformal in 30 lines.

No external data needed. Runs in < 2 seconds.
"""

import numpy as np
from conformal_ops import DICA

# Setup: nurse staffing LP (d=20 patients, budget=12)
d, budget = 20, 12.0
A_eq = np.ones((1, d))
b_eq = np.array([budget])
bounds = [(0.1, 1.0)] * d

# Run DICA (beta=0.5) and UCA (beta=0, standard conformal)
rng = np.random.RandomState(42)
for name, beta in [("UCA (standard)", 0.0), ("DICA (beta=0.5)", 0.5)]:
    model = DICA(alpha=0.10, beta=beta)
    for t in range(300):
        # Synthetic LOS: true = base + noise, pred = base + smaller noise
        base = rng.exponential(2.0, size=d)
        c_true = 0.5 + base / 10
        c_pred = 0.5 + (base + rng.normal(0, 0.5, d)) / 10

        result = model.step(c_pred, c_true, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

    stats = model.get_results()
    print(f"{name:20s}  PoC: {stats['avg_poc']:+.1%}  "
          f"Coverage: {stats['coverage']:.1%}  "
          f"DICA-cov: {stats['dica_coverage']:.1%}")
