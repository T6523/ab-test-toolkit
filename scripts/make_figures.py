"""Regenerate every figure and table in figures/. Fixed seeds, so reruns reproduce.

Run from the repo root: python scripts/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np

from abkit.planning import sample_size_per_arm
from abkit.sim import analyze_batch, power_curve, simulate_ab

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


def make_low_power():
    # Takeaway (for the README): at 2,000 users per arm power is only 18%, and the runs that
    # reach significance average +2.39 pp, 2.4x the true +1 pp, because only large positive
    # noise draws cross the threshold. At the planned 14,749 per arm power is 80% and the
    # significant runs average +1.13 pp, so planning n properly removes most of the exaggeration.
    true_lift, n_sims = 0.01, 100_000
    panels = [(2000, 4), (sample_size_per_arm(P_CONTROL, true_lift), 5)]  # (n per arm, seed)
    bins = np.arange(-3.0, 5.0, 0.1)  # same bins in both panels so the shapes compare

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
    for ax, (n, seed) in zip(axes, panels):
        sims = simulate_ab(n, P_CONTROL, true_lift, n_sims, seed=seed)
        res = analyze_batch(sims["x_control"], sims["x_treat"], n)
        lift_pp, sig = res["lift"] * 100, res["significant"]
        ax.hist(lift_pp, bins=bins, color="tab:gray", alpha=0.6,
                label=f"all {n_sims:,} runs")
        ax.hist(lift_pp[sig], bins=bins, color="tab:red", alpha=0.8,
                label=f"significant only ({sig.mean():.0%} of runs)")
        ax.axvline(true_lift * 100, color="black", linestyle="--", linewidth=1.5,
                   label=f"true lift = +{true_lift * 100:g} pp")
        ax.axvline(lift_pp[sig].mean(), color="tab:red", linestyle=":", linewidth=1.5,
                   label=f"mean of significant = {lift_pp[sig].mean():+.2f} pp "
                         f"({lift_pp[sig].mean() / (true_lift * 100):.2f}x true)")
        ax.set_title(f"n = {n:,} per arm (power {sig.mean():.0%})", fontsize=10)
        ax.set_xlabel("Estimated lift (pp)")
    axes[0].set_ylabel("Number of simulated tests")
    axes[0].set_ylim(top=axes[0].get_ylim()[1] * 1.35)  # headroom so legends clear the bars
    for ax in axes:
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Low power: significant results overstate the lift "
                 f"(true lift +{true_lift * 100:g} pp, baseline {P_CONTROL:.0%})", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "low_power.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIG_DIR.mkdir(exist_ok=True)
    make_power_curve()
    make_low_power()
    print(f"wrote figures to {FIG_DIR}")
