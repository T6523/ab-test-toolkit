"""Command line interface: `python -m abkit plan` and `python -m abkit analyze`."""
import argparse
import sys

import pandas as pd

from abkit.analysis import lift_ci, ztest_two_prop
from abkit.decide import decide, explain
from abkit.planning import sample_size_per_arm
from abkit.srm import srm_check

SRM_WARNING = ("WARNING: arm sizes differ from the planned split. "
               "Do not trust these results until the cause is found.")


def _fmt_p(p):
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def _pp(x):
    return f"{x * 100:+.2f} pp"


def cmd_plan(args):
    n = sample_size_per_arm(args.baseline, args.mde, alpha=args.alpha, power=args.power)
    print(f"Baseline conversion:      {args.baseline:.2%}")
    print(f"Minimum detectable lift:  {_pp(args.mde)} ({args.mde / args.baseline:+.1%} relative)")
    print(f"Significance level:       alpha = {args.alpha:g} (two-sided)")
    print(f"Power:                    {args.power:.0%}")
    print(f"Required sample size:     {n:,} per arm ({2 * n:,} total)")


def _load_counts(path):
    """Return (n_control, x_control, n_treat, x_treat) from a CSV file."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise ValueError(f"file not found: {path}")
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        raise ValueError(f"could not read {path}: {e}")

    for col in ("variant", "outcome"):
        if col not in df.columns:
            raise ValueError(
                f"column '{col}' not found (found: {', '.join(map(str, df.columns))})")
    found = sorted(map(str, df["variant"].unique()))
    if not set(df["variant"].unique()) <= {"control", "treatment"}:
        raise ValueError(
            f"variant must be 'control' or 'treatment' (found: {', '.join(found)})")
    if not set(df["outcome"].unique()) <= {0, 1}:
        raise ValueError(
            f"outcome must be 0 or 1 (found: {', '.join(map(str, df['outcome'].unique()))})")

    control = df[df["variant"] == "control"]["outcome"]
    treat = df[df["variant"] == "treatment"]["outcome"]
    if control.empty or treat.empty:
        raise ValueError("both 'control' and 'treatment' need at least one row")
    return len(control), int(control.sum()), len(treat), int(treat.sum())


def cmd_analyze(args):
    n_c, x_c, n_t, x_t = _load_counts(args.csv)
    srm = srm_check(n_c, n_t, expected_ratio=args.expected_ratio)
    test = ztest_two_prop(x_c, n_c, x_t, n_t)
    low, high = lift_ci(x_c, n_c, x_t, n_t, alpha=args.alpha)
    verdict = decide(low, high, args.delta)

    rows = [("Arm", "Users", "Conversions", "Rate"),
            ("control", f"{n_c:,}", f"{x_c:,}", f"{x_c / n_c:.2%}"),
            ("treatment", f"{n_t:,}", f"{x_t:,}", f"{x_t / n_t:.2%}")]
    widths = [max(len(r[i]) for r in rows) + 2 for i in range(3)]
    for r in rows:
        print("".join(c.ljust(w) for c, w in zip(r, widths)) + r[3])

    ratio = args.expected_ratio * 100
    status = "MISMATCH" if srm["flag"] else "OK"
    print()
    print(f"SRM check (expected {ratio:g}/{100 - ratio:g}): "
          f"chi2 = {srm['chi2']:.3f}, p = {_fmt_p(srm['p_value'])} -> {status}")
    if srm["flag"]:
        print(SRM_WARNING)

    rel = f"{test['lift'] / (x_c / n_c):+.1%} relative" if x_c else "relative lift n/a"
    print(f"Lift:     {_pp(test['lift'])} ({rel})")
    print(f"p-value:  {_fmt_p(test['p_value'])} (two-sided z-test, pooled SE)")
    print(f"{1 - args.alpha:.0%} CI:   [{_pp(low)}, {_pp(high)}]")
    print(f"Delta:    {args.delta * 100:.2f} pp")
    print(f"Verdict:  {verdict.name}")
    print(f"          {explain(verdict)}")
    if srm["flag"]:
        print(SRM_WARNING)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="abkit", description="Plan and analyze A/B tests on a binary conversion metric.")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="required sample size for a test")
    plan.add_argument("--baseline", type=float, required=True,
                      help="control conversion rate as a proportion, e.g. 0.10")
    plan.add_argument("--mde", type=float, required=True,
                      help="minimum detectable absolute lift as a proportion, e.g. 0.01")
    plan.add_argument("--alpha", type=float, default=0.05)
    plan.add_argument("--power", type=float, default=0.8)
    plan.set_defaults(func=cmd_plan)

    analyze = sub.add_parser("analyze", help="analyze a CSV of variant,outcome rows")
    analyze.add_argument("csv", help="CSV with columns 'variant' (control/treatment) and 'outcome' (0/1)")
    analyze.add_argument("--delta", type=float, required=True,
                         help="smallest absolute lift worth shipping, e.g. 0.005")
    analyze.add_argument("--alpha", type=float, default=0.05)
    analyze.add_argument("--expected-ratio", type=float, default=0.5,
                         help="planned control share of traffic")
    analyze.set_defaults(func=cmd_analyze)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
