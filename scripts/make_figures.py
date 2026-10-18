"""Regenerate every figure and table in figures/. Fixed seeds, so reruns reproduce.

Run from the repo root: python scripts/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from abkit.bayes import compare
from abkit.planning import sample_size_per_arm
from abkit.sequential import calibrate_obf_c, obf_boundaries, peeking_fp, sequential_fp
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


def make_peeking():
    looks = [1, 2, 5, 10, 20]
    n_total, n_sims = 10_000, 20_000
    rates = [peeking_fp(n_total, P_CONTROL, k, n_sims, seed=6) for k in looks]

    print(f"\nPeeking, A/A data, {n_total:,} users per arm in total, {n_sims:,} simulations")
    print("| Looks | False-positive rate |")
    print("|------:|--------------------:|")
    for k, r in zip(looks, rates):
        print(f"| {k:>5} | {r:>19.3f} |")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(looks, rates, "o-", color="tab:red")
    for k, r in zip(looks, rates):
        ax.annotate(f"{r:.1%}\n({k} look{'s' if k > 1 else ''})", (k, r),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.axhline(0.05, color="gray", linestyle="--", linewidth=1, label="nominal 5%")
    ax.set_xticks(looks)
    ax.set_xlabel("Number of looks (equally spaced)")
    ax.set_ylabel(f"False-positive rate ({n_sims:,} A/A runs per point)")
    ax.set_title("Peeking inflates false positives (z-test at every look, alpha 0.05)",
                 fontsize=10)
    ax.set_ylim(0, max(rates) * 1.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "peeking.png", dpi=150)
    plt.close(fig)


def make_sequential():
    looks = [1, 2, 5, 10, 20]
    n_total, n_sims = 10_000, 20_000
    cs = {k: calibrate_obf_c(k, n_sims=200_000, seed=7) for k in looks}
    peek = [peeking_fp(n_total, P_CONTROL, k, n_sims, seed=6) for k in looks]
    seq = [sequential_fp(n_total, P_CONTROL, k, cs[k], n_sims, seed=6) for k in looks]

    print(f"\nSequential boundaries, A/A data, {n_total:,} users per arm in total, {n_sims:,} simulations")
    print("| Looks | c | Peeking at 1.96 | Sequential |")
    print("|------:|------:|----------------:|-----------:|")
    for k, c, a, b in zip(looks, cs.values(), peek, seq):
        print(f"| {k:>5} | {c:.3f} | {a:>15.3f} | {b:>10.3f} |")
    for k in (5, 10):
        print(f"per-look critical z, K = {k}: {[round(z, 2) for z in obf_boundaries(k, cs[k])]}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(looks, peek, "o-", color="tab:red", label="peeking (z > 1.96 at every look)")
    ax.plot(looks, seq, "s-", color="tab:blue", label="sequential (calibrated boundaries)")
    for k, a, b in zip(looks, peek, seq):
        ax.annotate(f"{a:.1%}\n({k} look{'s' if k > 1 else ''})", (k, a), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="tab:red")
        ax.annotate(f"{b:.1%}", (k, b), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color="tab:blue")
    ax.axhline(0.05, color="gray", linestyle="--", linewidth=1, label="nominal 5%")
    ax.set_xticks(looks)
    ax.set_xlabel("Number of looks (equally spaced)")
    ax.set_ylabel(f"False-positive rate ({n_sims:,} A/A runs per point)")
    ax.set_title("Sequential boundaries keep false positives at 5% (alpha 0.05 two-sided)", fontsize=10)
    ax.set_ylim(0, max(peek) * 1.25)
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sequential.png", dpi=150)
    plt.close(fig)

    # power cost: true lift 0.01 at the planned n (10 equally spaced looks)
    n_plan, lift, n_sims_power = sample_size_per_arm(P_CONTROL, 0.01), 0.01, 100_000
    fixed = peeking_fp(n_plan, P_CONTROL, 1, n_sims_power, seed=8, true_lift=lift)
    peek10 = peeking_fp(n_plan, P_CONTROL, 10, n_sims_power, seed=8, true_lift=lift)
    seq10 = sequential_fp(n_plan, P_CONTROL, 10, cs[10], n_sims_power, seed=8, true_lift=lift)
    print(f"\nPower, true lift +{lift * 100:g} pp, baseline {P_CONTROL:.0%}, {n_plan:,} users per arm, {n_sims_power:,} simulations")
    print("| Rule | Power |")
    print("|------|------:|")
    print(f"| single fixed-horizon test | {fixed:.3f} |")
    print(f"| sequential, 10 looks | {seq10:.3f} |")
    print(f"| peeking at 1.96, 10 looks (false positives {peek[looks.index(10)]:.1%}) | {peek10:.3f} |")
    print(f"sequential power cost vs fixed horizon: {fixed - seq10:.3f}")


def make_bayes_vs_freq():
    # DRAFT takeaway for the README (owner to edit). The rules disagree on borderline results: a positive
    # lift that is not significant (z between about 1 and 1.96) is often small enough in expected loss for the
    # Bayesian rule to ship, so at 2,000 users per arm it ships 11% of no-effect runs and 42% of +1 pp runs,
    # against 2% and 18% for the frequentist rule. The Bayesian rule does not control false positives at 5%;
    # it controls the average conversion rate given up when it ships (expected loss below epsilon), which is
    # the more useful question when a wrong launch costs little. Limitation: the uniform prior and
    # epsilon = 0.05 pp are assumptions, and a different epsilon moves the ship rates a lot (at
    # epsilon = 0.01 pp the Bayesian rule is about as strict as the frequentist one).
    n, n_sims, draws = 2000, 4000, 20_000
    epsilons = (0.0005, 0.0001)  # 0.05 pp and 0.01 pp
    rows = []
    for lift, seed in ((0.0, 11), (0.01, 12)):
        sims = simulate_ab(n, P_CONTROL, lift, n_sims, seed=seed)
        res = analyze_batch(sims["x_control"], sims["x_treat"], n)
        freq = res["significant"] & (res["lift"] > 0)  # p < 0.05 and treatment better
        loss = np.array([compare(int(xc), n, int(xt), n, draws=draws, seed=i)["expected_loss_choose_b"]
                         for i, (xc, xt) in enumerate(zip(sims["x_control"], sims["x_treat"]))])
        bayes = [loss < eps for eps in epsilons]  # same posterior draws, two thresholds
        rows.append((lift, freq.mean(), [b.mean() for b in bayes], [(freq == b).mean() for b in bayes]))

    # z cutoff each Bayesian rule is equivalent to (normal approximation of the posterior of the lift):
    # expected loss = se * (pdf(z) - z * sf(z)) with z = lift / se, solved for the epsilon threshold
    se = np.sqrt(2 * P_CONTROL * (1 - P_CONTROL) / n)
    z_cut = [brentq(lambda z: se * (norm.pdf(z) - z * norm.sf(z)) - eps, 0, 6) for eps in epsilons]

    print(f"\nShip-B rates, {n:,} users per arm, baseline {P_CONTROL:.0%}, {n_sims:,} simulations per scenario")
    print("| True lift | Frequentist (p < 0.05, lift > 0) | Bayesian eps 0.05 pp | Bayesian eps 0.01 pp | Agreement with frequentist (0.05 pp / 0.01 pp) |")
    print("|----------:|---------------------------------:|---------------------:|---------------------:|-----------------------------------------------:|")
    for lift, f, b, agree in rows:
        print(f"| {lift * 100:+.2f} pp | {f:.3f} | {b[0]:.3f} | {b[1]:.3f} | {agree[0]:.3f} / {agree[1]:.3f} |")
    print(f"z cutoffs: frequentist 1.96, Bayesian eps 0.05 pp {z_cut[0]:.2f}, eps 0.01 pp {z_cut[1]:.2f}")

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = [([r[1] for r in rows], "tab:blue", "frequentist: p < 0.05 and lift > 0"),
              ([r[2][0] for r in rows], "tab:orange", "Bayesian: expected loss < 0.05 pp"),
              ([r[2][1] for r in rows], "tab:green", "Bayesian: expected loss < 0.01 pp")]
    for k, (vals, color, label) in enumerate(groups):
        bars = ax.bar(x + (k - 1) * 0.27, vals, 0.27, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in vals], padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"true lift {r[0] * 100:+g} pp\nagrees with frequentist on\n"
                        f"{r[3][0]:.1%} (eps 0.05 pp), {r[3][1]:.1%} (eps 0.01 pp) of runs" for r in rows], fontsize=8)
    ax.set_ylabel(f"Fraction of {n_sims:,} simulations that ship B")
    ax.set_title(f"Bayesian vs frequentist ship rate ({n:,} users per arm, baseline {P_CONTROL:.0%})",
                 fontsize=10)
    ax.set_ylim(0, max(max(r[1], *r[2]) for r in rows) * 1.2)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "bayes_vs_freq.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIG_DIR.mkdir(exist_ok=True)
    make_power_curve()
    make_low_power()
    make_peeking()
    make_sequential()
    make_bayes_vs_freq()
    print(f"wrote figures to {FIG_DIR}")
