"""Regenerate every figure and table in figures/. Fixed seeds, so reruns reproduce.

Run from the repo root: python scripts/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np

from abkit.planning import sample_size_per_arm
from abkit.sim import power_curve

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
P_CONTROL = 0.10


def make_power_curve():
    lifts = [0.005, 0.01, 0.02]
    grid = np.unique(np.geomspace(500, 60_000, 30).round().astype(int)).tolist()
    curves = power_curve(P_CONTROL, lifts, grid, n_sims=20_000, seed=3)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lift, powers in curves.items():
        line, = ax.plot(grid, powers, label=f"lift = +{lift * 100:g} pp")
        planned = sample_size_per_arm(P_CONTROL, lift)
        ax.plot(planned, 0.8, "o", color=line.get_color())
        ax.annotate(f"{planned:,}", (planned, 0.8), textcoords="offset points",
                    xytext=(5, -14), fontsize=8, color=line.get_color())
    ax.axhline(0.8, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Users per arm")
    ax.set_ylabel("Power (simulated, 20,000 runs per point)")
    ax.set_title(f"Power vs sample size (baseline {P_CONTROL:.0%}, alpha 0.05 two-sided)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "power_curve.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIG_DIR.mkdir(exist_ok=True)
    make_power_curve()
    print(f"wrote {FIG_DIR / 'power_curve.png'}")
