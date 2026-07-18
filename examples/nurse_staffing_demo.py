"""
Full nurse staffing demo with visualization.

Compares DICA vs UCA vs Nominal over 500 rounds with synthetic
distribution shift. Generates three plots in examples/output/.
"""

import numpy as np
import os
from conformal_ops import DICA

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not installed — skipping plots. Install with: pip install matplotlib")


def generate_data(T: int, d: int, seed: int = 42, shift_at: int = 250):
    """Generate synthetic LOS data with distribution shift at round `shift_at`."""
    rng = np.random.RandomState(seed)
    data = []
    for t in range(T):
        # Base LOS: exponential, with shift in mean after `shift_at`
        mean = 2.0 if t < shift_at else 3.5  # distribution shift
        base = rng.exponential(mean, size=d)
        noise = rng.normal(0, 0.5, size=d)
        los_true = np.maximum(base, 0.1)
        los_pred = np.maximum(base + noise, 0.1)
        c_true = 0.5 + los_true / 10
        c_pred = 0.5 + los_pred / 10
        data.append((c_pred, c_true))
    return data


def run_experiment():
    T, d = 500, 20
    budget = 12.0
    A_eq = np.ones((1, d))
    b_eq = np.array([budget])
    bounds = [(0.1, 1.0)] * d

    data = generate_data(T, d)

    results = {}
    for name, beta in [("UCA", 0.0), ("DICA", 0.5)]:
        model = DICA(alpha=0.10, beta=beta)
        round_data = []
        for t, (c_pred, c_true) in enumerate(data):
            r = model.step(c_pred, c_true, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
            round_data.append(r)
        results[name] = {
            "rounds": round_data,
            "stats": model.get_results(),
            "poc_history": model.poc_history,
            "model": model,
        }

    # Print summary
    print(f"{'Method':10s}  {'PoC':>8s}  {'Coverage':>10s}  {'DICA-cov':>10s}")
    print("-" * 45)
    for name, res in results.items():
        s = res["stats"]
        print(f"{name:10s}  {s['avg_poc']:+7.1%}  {s['coverage']:>9.1%}  {s['dica_coverage']:>9.1%}")

    # Plots
    if HAS_MPL:
        os.makedirs("examples/output", exist_ok=True)

        # 1. Coverage trajectory
        fig, ax = plt.subplots(figsize=(10, 4))
        w = 50
        for name, color in [("UCA", "gray"), ("DICA", "#2196F3")]:
            covs = [r["std_covered"] for r in results[name]["rounds"]]
            rolling = np.convolve(covs, np.ones(w)/w, mode="valid")
            ax.plot(rolling, label=name, color=color, linewidth=1.5)
        ax.axhline(0.9, color="red", linestyle="--", alpha=0.5, label="Target (90%)")
        ax.axvline(250, color="orange", linestyle=":", alpha=0.5, label="Distribution shift")
        ax.set_xlabel("Round")
        ax.set_ylabel("Rolling Coverage (w=50)")
        ax.set_title("Coverage Trajectory: DICA vs UCA")
        ax.legend()
        ax.set_ylim(0.6, 1.05)
        fig.tight_layout()
        fig.savefig("examples/output/coverage_trajectory.png", dpi=150)
        print("Saved: examples/output/coverage_trajectory.png")

        # 2. PoC over time
        fig, ax = plt.subplots(figsize=(10, 4))
        for name, color in [("UCA", "gray"), ("DICA", "#2196F3")]:
            pocs = results[name]["poc_history"]
            rolling = np.convolve(pocs, np.ones(w)/w, mode="valid")
            ax.plot(rolling * 100, label=name, color=color, linewidth=1.5)
        ax.axvline(250, color="orange", linestyle=":", alpha=0.5, label="Distribution shift")
        ax.set_xlabel("Round")
        ax.set_ylabel("Price of Coverage (%)")
        ax.set_title("PoC Trajectory: DICA vs UCA")
        ax.legend()
        fig.tight_layout()
        fig.savefig("examples/output/poc_trajectory.png", dpi=150)
        print("Saved: examples/output/poc_trajectory.png")

        # 3. Radii comparison (last round)
        fig, ax = plt.subplots(figsize=(8, 4))
        last_round_dica = results["DICA"]["rounds"][-1]["radii"]
        last_round_uca = results["UCA"]["rounds"][-1]["radii"]
        x = np.arange(d)
        ax.bar(x - 0.2, last_round_uca, 0.4, label="UCA (uniform)", color="gray", alpha=0.7)
        ax.bar(x + 0.2, last_round_dica, 0.4, label="DICA (reshaped)", color="#2196F3", alpha=0.7)
        ax.set_xlabel("Patient dimension j")
        ax.set_ylabel("Conformal radius r_j")
        ax.set_title("Radii Redistribution: DICA vs UCA (last round)")
        ax.legend()
        fig.tight_layout()
        fig.savefig("examples/output/radii_comparison.png", dpi=150)
        print("Saved: examples/output/radii_comparison.png")

        plt.close("all")


if __name__ == "__main__":
    run_experiment()
