import numpy as np
import pytest

from abkit.analysis import lift_ci, ztest_two_prop
from abkit.planning import power_at_n, sample_size_per_arm
from abkit.sim import (analyze_batch, ci_coverage, exaggeration_stats,
                       false_positive_rate, power_curve, simulate_ab)

N_SIMS = 20000  # Monte Carlo standard error is about 0.0015 for rates near 5%


# simulate_ab

def test_simulate_ab_shape_dtype_and_reproducible():
    a = simulate_ab(2000, 0.10, 0.01, 5, seed=1)
    b = simulate_ab(2000, 0.10, 0.01, 5, seed=1)
    assert set(a) == {"x_control", "x_treat"}
    for key in a:
        assert a[key].shape == (5,) and np.issubdtype(a[key].dtype, np.integer)
        assert (a[key] == b[key]).all()


def test_simulate_ab_different_seed_differs():
    a = simulate_ab(2000, 0.10, 0.01, 50, seed=1)
    b = simulate_ab(2000, 0.10, 0.01, 50, seed=2)
    assert not (a["x_control"] == b["x_control"]).all()


def test_simulate_ab_rates_match_truth():
    r = simulate_ab(10000, 0.10, 0.02, 50000, seed=0)
    assert r["x_control"].mean() / 10000 == pytest.approx(0.10, abs=0.0005)
    assert r["x_treat"].mean() / 10000 == pytest.approx(0.12, abs=0.0005)


@pytest.mark.parametrize("args", [
    (100, 0.99, 0.02, 10, 0),   # treatment rate 1.01
    (100, 0.0, 0.01, 10, 0),    # p_control = 0
    (100, 0.10, -0.10, 10, 0),  # treatment rate 0
    (0, 0.10, 0.01, 10, 0),     # no users
    (100, 0.10, 0.01, 0, 0),    # no simulations
    (100.5, 0.10, 0.01, 10, 0),
])
def test_simulate_ab_invalid_raises(args):
    with pytest.raises(ValueError):
        simulate_ab(*args)


# analyze_batch

def test_analyze_batch_known_values():
    r = analyze_batch([100, 100, 120], [110, 100, 100], 1000)
    assert r["lift"] == pytest.approx([0.01, 0.0, -0.02])
    assert r["z"] == pytest.approx([0.729422, 0.0, -1.429301], abs=1e-6)
    assert r["p_value"] == pytest.approx([0.465744, 1.0, 0.152918], abs=1e-6)
    assert r["low"] == pytest.approx([-0.016867, -0.026296, -0.047411], abs=1e-6)
    assert r["high"] == pytest.approx([0.036867, 0.026296, 0.007411], abs=1e-6)
    assert not r["significant"].any()


def test_analyze_batch_significant_flags():
    r = analyze_batch([100, 200], [150, 200], 1000)
    assert r["z"] == pytest.approx([3.380617, 0.0], abs=1e-6)
    assert r["significant"].tolist() == [True, False]


def test_analyze_batch_equals_scalar_functions():
    rng = np.random.default_rng(0)
    xc = rng.binomial(1000, 0.10, 300)
    xt = rng.binomial(1000, 0.12, 300)
    batch = analyze_batch(xc, xt, 1000)
    for i in range(300):
        scalar = ztest_two_prop(int(xc[i]), 1000, int(xt[i]), 1000)
        low, high = lift_ci(int(xc[i]), 1000, int(xt[i]), 1000)
        assert batch["lift"][i] == pytest.approx(scalar["lift"], rel=1e-12, abs=1e-15)
        assert batch["z"][i] == pytest.approx(scalar["z"], rel=1e-12, abs=1e-15)
        assert batch["p_value"][i] == pytest.approx(scalar["p_value"], rel=1e-12, abs=1e-15)
        assert batch["low"][i] == pytest.approx(low, rel=1e-12, abs=1e-15)
        assert batch["high"][i] == pytest.approx(high, rel=1e-12, abs=1e-15)


def test_analyze_batch_pooled_rate_zero_or_one():
    r = analyze_batch([0, 100], [0, 100], 100)
    assert r["z"].tolist() == [0.0, 0.0]
    assert r["p_value"].tolist() == [1.0, 1.0]
    assert not r["significant"].any()


@pytest.mark.parametrize("kwargs", [
    {"x_control": [1, 2], "x_treat": [10], "n_per_arm": 1000},      # shape mismatch
    {"x_control": [5], "x_treat": [5], "n_per_arm": 0},
    {"x_control": [5.5], "x_treat": [5], "n_per_arm": 100},         # non-integer
    {"x_control": [1100], "x_treat": [5], "n_per_arm": 1000},       # x > n
    {"x_control": [-1], "x_treat": [5], "n_per_arm": 1000},         # x < 0
    {"x_control": [1], "x_treat": [5], "n_per_arm": 1000, "alpha": 0},
    {"x_control": [1], "x_treat": [5], "n_per_arm": 1000, "alpha": 1},
])
def test_analyze_batch_invalid_raises(kwargs):
    with pytest.raises(ValueError):
        analyze_batch(**kwargs)


# calibration: do the methods deliver the error rates they claim?

def test_false_positive_rate_is_alpha():
    # A/A at n = 5,000: reference value 0.0516, band is about +/- 3 SE
    assert 0.045 <= false_positive_rate(5000, 0.10, N_SIMS, seed=1) <= 0.055


def test_false_positive_rate_follows_alpha():
    assert false_positive_rate(5000, 0.10, N_SIMS, seed=1, alpha=0.01) == pytest.approx(0.01, abs=0.004)


def test_ci_coverage_is_95_percent():
    assert 0.94 <= ci_coverage(5000, 0.10, 0.01, N_SIMS, seed=2) <= 0.96


def test_ci_coverage_follows_alpha():
    assert 0.985 <= ci_coverage(5000, 0.10, 0.01, N_SIMS, seed=2, alpha=0.01) <= 0.995


def test_simulated_power_matches_formula():
    lifts, grid = [0.005, 0.01, 0.02], [2000, 14749, 50000]
    curve = power_curve(0.10, lifts, grid, N_SIMS, seed=3)
    for lift in lifts:
        for n, simulated in zip(grid, curve[lift]):
            assert simulated == pytest.approx(power_at_n(0.10, lift, n), abs=0.02)


def test_power_curve_crosses_80_percent_at_planned_n():
    planned = sample_size_per_arm(0.10, 0.01)
    grid = [round(planned * f) for f in (0.90, 0.95, 1.00, 1.05, 1.10)]
    powers = power_curve(0.10, [0.01], grid, N_SIMS, seed=9)[0.01]
    # interpolate the first crossing of 0.80
    i = next(k for k, p in enumerate(powers) if p >= 0.80)
    assert i > 0
    cross = grid[i - 1] + (0.80 - powers[i - 1]) / (powers[i] - powers[i - 1]) * (grid[i] - grid[i - 1])
    assert abs(cross - planned) / planned <= 0.05


def test_power_curve_is_increasing_in_n_and_lift():
    curve = power_curve(0.10, [0.005, 0.01], [1000, 5000, 20000], N_SIMS, seed=4)
    for powers in curve.values():
        assert powers == sorted(powers)
    assert all(a < b for a, b in zip(curve[0.005], curve[0.01]))


# exaggeration_stats (type M and type S errors)

def test_exaggeration_underpowered():
    r = exaggeration_stats(2000, 0.10, 0.01, 100000, seed=4)
    assert r["power"] == pytest.approx(0.176, abs=0.01)
    assert r["exaggeration_ratio"] == pytest.approx(2.42, abs=0.1)
    assert r["type_s"] == pytest.approx(0.0075, abs=0.003)
    assert r["n_significant"] == pytest.approx(r["power"] * 100000, abs=1)


def test_exaggeration_well_powered():
    r = exaggeration_stats(14749, 0.10, 0.01, 100000, seed=5)
    assert r["power"] == pytest.approx(0.80, abs=0.01)
    assert r["exaggeration_ratio"] == pytest.approx(1.12, abs=0.03)
    assert r["type_s"] == 0.0


def test_exaggeration_negative_lift_mirrors_positive():
    pos = exaggeration_stats(2000, 0.10, 0.01, 100000, seed=4)
    neg = exaggeration_stats(2000, 0.10, -0.01, 100000, seed=4)
    assert neg["exaggeration_ratio"] > 2
    assert neg["type_s"] < 0.02 and pos["type_s"] < 0.02


def test_exaggeration_no_significant_runs_warns():
    with pytest.warns(UserWarning, match="no significant"):
        r = exaggeration_stats(10, 0.50, 0.0001, 5, seed=0)
    assert r["n_significant"] == 0
    assert np.isnan(r["exaggeration_ratio"]) and np.isnan(r["type_s"])


def test_exaggeration_zero_lift_raises():
    with pytest.raises(ValueError):
        exaggeration_stats(2000, 0.10, 0, 1000, seed=1)
