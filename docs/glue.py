import json
import numpy as np
from scipy.stats import norm
from abkit.planning import sample_size_per_arm, power_at_n
from abkit.analysis import ztest_two_prop, lift_ci, relative_lift_ci
from abkit.decide import decide, explain
from abkit.srm import srm_check
from abkit.sim import simulate_ab, analyze_batch, exaggeration_stats, get_rng
from abkit.sequential import peeking_fp, sequential_fp, calibrate_obf_c, obf_boundaries

_c_cache = {}

# Inputs the page shows on load. scripts/build_site.py runs these to write docs/initial.json,
# which the page draws immediately while Python is still loading. Keep in sync with the form defaults in index.html.
DEFAULTS = {
    "plan": (0.1, 0.01, 0.05, 0.8),
    "analyze": (10000, 1000, 10000, 1100, 0.005, 0.05, 0.5),
    "peek": (10000, 0.1, 5000),
    "path": (10000, 0.1, 10, 1, True),
    "low": (2000, 0.1, 0.01),
}


def _int(name, v):
    if v != int(v):
        raise ValueError(f"{name} must be a whole number, got {v}")
    return int(v)


def _num(x):
    x = float(x)
    return None if x != x else x


def _c(k):
    if k not in _c_cache:
        _c_cache[k] = calibrate_obf_c(k, n_sims=100000, seed=7)
    return _c_cache[k]


def plan(baseline, mde, alpha, power):
    n = sample_size_per_arm(baseline, mde, alpha=alpha, power=power)
    lifts = []
    for f in (0.5, 1.0, 2.0):
        lift = mde * f
        if 0 < baseline + lift < 1:
            lifts.append(lift)
    n_max = sample_size_per_arm(baseline, min(lifts, key=abs), alpha=alpha, power=power)
    n_max = int(max(n * 1.3, n_max * 1.2))
    grid = sorted(set(int(v) for v in np.geomspace(max(20, n_max / 200), n_max, 60)))
    curves = [{"lift": lift, "powers": [power_at_n(baseline, lift, g, alpha=alpha) for g in grid]} for lift in lifts]
    return json.dumps({"n": n, "grid": grid, "curves": curves, "mde": mde, "power": power})


def analyze(nc, xc, nt, xt, delta, alpha, ratio):
    nc, xc, nt, xt = _int("control users", nc), _int("control conversions", xc), _int("treatment users", nt), _int("treatment conversions", xt)
    test = ztest_two_prop(xc, nc, xt, nt)
    low, high = lift_ci(xc, nc, xt, nt, alpha=alpha)
    rel = None
    if xc > 0:
        rlow, rhigh = relative_lift_ci(xc, nc, xt, nt, alpha=alpha)
        rel = test["lift"] / (xc / nc)
    verdict = decide(low, high, delta)
    srm = srm_check(nc, nt, expected_ratio=ratio)
    return json.dumps({"lift": test["lift"], "rel": rel, "p": test["p_value"], "z": test["z"], "low": low, "high": high,
                       "delta": delta, "verdict": verdict.name, "why": explain(verdict), "srm": srm,
                       "rates": [xc / nc, xt / nt]})


def peeking(n_total, baseline, sims):
    n_total, sims = _int("users", n_total), _int("simulations", sims)
    looks = [1, 2, 5, 10, 20]
    rows = []
    for k in looks:
        c = _c(k)
        rows.append({"looks": k, "c": c,
                     "peeking": peeking_fp(n_total, baseline, k, sims, seed=6),
                     "sequential": sequential_fp(n_total, baseline, k, c, sims, seed=6)})
    return json.dumps({"rows": rows})


def _path(n_total, baseline, k, seed):
    step = n_total // k
    rng = get_rng(seed)
    xc = xt = 0
    z = []
    for i in range(1, k + 1):
        xc += int(rng.binomial(step, baseline))
        xt += int(rng.binomial(step, baseline))
        z.append(abs(ztest_two_prop(xc, step * i, xt, step * i)["z"]))
    bounds = obf_boundaries(k, _c(k))
    peek_hit = next((i + 1 for i, v in enumerate(z) if v > norm.ppf(0.975)), None)
    seq_hit = next((i + 1 for i, v in enumerate(z) if v >= bounds[i]), None)
    return z, bounds, peek_hit, seq_hit


def path(n_total, baseline, k, seed, fooled):
    n_total, k, seed = _int("users", n_total), _int("looks", k), _int("seed", seed)
    if fooled:
        for s in range(seed, seed + 5000):
            z, bounds, peek_hit, seq_hit = _path(n_total, baseline, k, s)
            if peek_hit is not None and peek_hit >= min(3, k) and seq_hit is None:
                seed = s
                break
    else:
        z, bounds, peek_hit, seq_hit = _path(n_total, baseline, k, seed)
    return json.dumps({"seed": seed, "k": k, "step": n_total // k, "z": z, "bounds": bounds,
                       "crit": float(norm.ppf(0.975)), "peek_hit": peek_hit, "seq_hit": seq_hit})


def lowpower(n, baseline, lift):
    n = _int("users", n)
    sims = simulate_ab(n, baseline, lift, 20000, seed=4)
    res = analyze_batch(sims["x_control"], sims["x_treat"], n)
    lift_pp, sig = res["lift"] * 100, res["significant"]
    s = float(np.sqrt(2 * baseline * (1 - baseline) / n)) * 100
    edges = np.linspace(lift * 100 - 6 * s, lift * 100 + 6 * s, 61)
    all_counts, _ = np.histogram(lift_pp, bins=edges)
    sig_counts, _ = np.histogram(lift_pp[sig], bins=edges)
    stats = exaggeration_stats(n, baseline, lift, 20000, seed=4)
    return json.dumps({"centers": ((edges[:-1] + edges[1:]) / 2).tolist(), "all": all_counts.tolist(),
                       "sig": sig_counts.tolist(), "lift_pp": lift * 100,
                       "mean_sig": _num(lift_pp[sig].mean()) if sig.any() else None,
                       "power": stats["power"], "ratio": _num(stats["exaggeration_ratio"]),
                       "type_s": _num(stats["type_s"]), "n_sig": stats["n_significant"]})
