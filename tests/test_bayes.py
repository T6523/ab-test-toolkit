import numpy as np
import pytest
from scipy.stats import norm

from abkit.analysis import lift_ci
from abkit.bayes import compare

FIXTURE_A = (1000, 10000, 1100, 10000)


@pytest.mark.parametrize("args", [FIXTURE_A, (100, 1000, 100, 1000), (10, 100, 12, 100), (0, 50, 50, 50)])
def test_posterior_means_are_exact(args):
    x_c, n_c, x_t, n_t = args
    r = compare(*args, draws=10)
    assert r["mean_control"] == (1 + x_c) / (2 + n_c)
    assert r["mean_treat"] == (1 + x_t) / (2 + n_t)


def test_posterior_means_reference_values():
    r = compare(*FIXTURE_A, draws=10)
    assert r["mean_control"] == pytest.approx(0.100080, abs=1e-6)
    assert r["mean_treat"] == pytest.approx(0.110078, abs=1e-6)


@pytest.mark.parametrize("args, p_beats, loss, ship", [
    (FIXTURE_A, 0.989, 0.00002, True),
    ((100, 1000, 100, 1000), 0.50, 0.0054, False),      # identical data
    ((10, 100, 12, 100), 0.67, 0.0097, False),          # small sample: higher rate, too thin to ship
    ((1100, 10000, 1000, 10000), 0.011, 0.0100, False), # treatment worse
])
def test_compare_reference_values(args, p_beats, loss, ship):
    r = compare(*args)
    assert r["p_b_beats_a"] == pytest.approx(p_beats, abs=0.01)
    assert r["expected_loss_choose_b"] == pytest.approx(loss, abs=0.0003)
    assert r["ship_b"] is ship


def test_p_b_beats_a_matches_normal_approximation_for_large_n():
    # large n: posterior of the difference is about N(lift, se_unpooled^2), so P(B > A) = Phi(lift / se)
    args = (5000, 50000, 5400, 50000)
    low, high = lift_ci(*args)
    lift, se = (low + high) / 2, (high - low) / (2 * norm.ppf(0.975))
    assert compare(*args, draws=200000)["p_b_beats_a"] == pytest.approx(norm.cdf(lift / se), abs=0.01)


def test_expected_loss_shrinks_with_more_data():
    small = compare(10, 100, 12, 100)["expected_loss_choose_b"]
    large = compare(1000, 10000, 1200, 10000)["expected_loss_choose_b"]
    assert large < small


def test_symmetry_swapping_arms():
    a = compare(*FIXTURE_A, draws=200000, seed=1)
    b = compare(1100, 10000, 1000, 10000, draws=200000, seed=1)
    assert a["p_b_beats_a"] + b["p_b_beats_a"] == pytest.approx(1.0, abs=0.005)


def test_epsilon_controls_ship_decision():
    loose = compare(10, 100, 12, 100, epsilon=0.05)
    strict = compare(10, 100, 12, 100, epsilon=0.0005)
    assert loose["ship_b"] is True and strict["ship_b"] is False
    assert loose["expected_loss_choose_b"] == strict["expected_loss_choose_b"]


def test_reproducible_and_seed_matters():
    assert compare(*FIXTURE_A, seed=3) == compare(*FIXTURE_A, seed=3)
    assert compare(*FIXTURE_A, seed=3)["p_b_beats_a"] != compare(*FIXTURE_A, seed=4)["p_b_beats_a"]


def test_return_types_are_plain_python():
    r = compare(*FIXTURE_A, draws=100)
    assert set(r) == {"mean_control", "mean_treat", "p_b_beats_a", "expected_loss_choose_b", "ship_b"}
    assert all(type(r[k]) is float for k in ("mean_control", "mean_treat", "p_b_beats_a", "expected_loss_choose_b"))
    assert type(r["ship_b"]) is bool


def test_accepts_numpy_integers():
    args = [np.int64(v) for v in FIXTURE_A]
    assert compare(*args, draws=100)["mean_control"] == pytest.approx(0.100080, abs=1e-6)


@pytest.mark.parametrize("args, kwargs", [
    ((10, 5, 1, 10), {}),            # x_control > n_control
    ((1, 10, 11, 10), {}),           # x_treat > n_treat
    ((-1, 10, 1, 10), {}),           # negative count
    ((1, 0, 1, 10), {}),             # n_control = 0
    ((1.5, 10, 1, 10), {}),          # non-integer count
    ((1, 10, 1, 10), {"draws": 0}),
    ((1, 10, 1, 10), {"draws": 1.5}),
    ((1, 10, 1, 10), {"epsilon": 0}),
    ((1, 10, 1, 10), {"epsilon": -0.001}),
])
def test_invalid_raises(args, kwargs):
    with pytest.raises(ValueError):
        compare(*args, **kwargs)
