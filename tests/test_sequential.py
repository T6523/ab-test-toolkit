import pytest
from scipy.stats import norm

from abkit.sequential import calibrate_obf_c, obf_boundaries, peeking_fp, sequential_fp
from abkit.sim import false_positive_rate

N_SIMS = 20000  # Monte Carlo standard error is about 0.0015 to 0.003 for these rates
N_TOTAL, P = 10000, 0.10


# peeking_fp

def test_peeking_one_look_is_the_ordinary_test():
    # same seed, same draw order as simulate_ab, so the result is identical
    assert peeking_fp(N_TOTAL, P, 1, N_SIMS, seed=1) == false_positive_rate(N_TOTAL, P, N_SIMS, seed=1)


def test_peeking_one_look_is_alpha():
    assert 0.045 <= peeking_fp(N_TOTAL, P, 1, N_SIMS, seed=1) <= 0.055


def test_peeking_ten_looks_inflates_to_about_19_percent():
    # reference 0.193 (spec 4.7, 7.5)
    assert 0.18 <= peeking_fp(N_TOTAL, P, 10, N_SIMS, seed=1) <= 0.21


@pytest.mark.parametrize("looks, expected", [(2, 0.081), (5, 0.139), (20, 0.245)])
def test_peeking_reference_values(looks, expected):
    assert peeking_fp(N_TOTAL, P, looks, N_SIMS, seed=1) == pytest.approx(expected, abs=0.012)


def test_peeking_increases_with_looks():
    rates = [peeking_fp(N_TOTAL, P, k, N_SIMS, seed=2) for k in (1, 2, 5, 10, 20)]
    assert rates == sorted(rates) and rates[0] < rates[-1]


def test_peeking_follows_alpha():
    assert peeking_fp(N_TOTAL, P, 1, N_SIMS, seed=1, alpha=0.01) == pytest.approx(0.01, abs=0.004)
    assert peeking_fp(N_TOTAL, P, 10, N_SIMS, seed=1, alpha=0.01) < peeking_fp(N_TOTAL, P, 10, N_SIMS, seed=1)


def test_peeking_reproducible_and_seed_matters():
    a = peeking_fp(N_TOTAL, P, 5, 2000, seed=3)
    assert a == peeking_fp(N_TOTAL, P, 5, 2000, seed=3)
    assert a != peeking_fp(N_TOTAL, P, 5, 2000, seed=4)


def test_peeking_not_divisible_uses_effective_total():
    # 10,001 users over 3 looks: step 3333, effective total 9999, still a valid rate above alpha
    assert 0.05 < peeking_fp(10001, P, 3, N_SIMS, seed=1) < 0.20


@pytest.mark.parametrize("args", [
    (0, P, 5, 100, 0),       # no users
    (100, P, 0, 100, 0),     # no looks
    (5, P, 10, 100, 0),      # more looks than users
    (100.5, P, 5, 100, 0),   # non-integer
    (100, P, 5, 0, 0),       # no simulations
    (100, 0.0, 5, 100, 0),   # p_control = 0
    (100, 1.0, 5, 100, 0),   # p_control = 1
    (100, P, 5, 100, 0, 0),  # alpha = 0
    (100, P, 5, 100, 0, 1),  # alpha = 1
])
def test_peeking_invalid_raises(args):
    with pytest.raises(ValueError):
        peeking_fp(*args)


# obf_boundaries

@pytest.mark.parametrize("looks, c, expected", [
    (5, 2.04, [4.562, 3.226, 2.634, 2.281, 2.040]),
    (10, 2.09, [6.609, 4.673, 3.816, 3.305, 2.956, 2.698, 2.498, 2.337, 2.203, 2.090]),
])
def test_obf_boundaries_reference_values(looks, c, expected):
    assert obf_boundaries(looks, c) == pytest.approx(expected, abs=1e-3)


def test_obf_boundaries_shape():
    b = obf_boundaries(8, 2.0)
    assert len(b) == 8 and b[-1] == pytest.approx(2.0)
    assert b == sorted(b, reverse=True)           # early looks are the strictest
    assert obf_boundaries(1, 1.96) == [1.96]      # one look: no correction


@pytest.mark.parametrize("args", [(0, 2.0), (-3, 2.0), (2.5, 2.0), (5, 0), (5, -1.0)])
def test_obf_boundaries_invalid_raises(args):
    with pytest.raises(ValueError):
        obf_boundaries(*args)


# calibrate_obf_c

def test_calibrate_one_look_is_normal_quantile():
    assert calibrate_obf_c(1, n_sims=200000, seed=7) == pytest.approx(norm.ppf(0.975), abs=0.03)


@pytest.mark.parametrize("looks, expected", [(5, 2.04), (10, 2.09)])
def test_calibrate_reference_values(looks, expected):
    assert calibrate_obf_c(looks, n_sims=200000, seed=7) == pytest.approx(expected, abs=0.03)


def test_calibrate_grows_with_looks_and_shrinks_with_alpha():
    c = [calibrate_obf_c(k, n_sims=100000, seed=1) for k in (1, 2, 5, 10)]
    assert c == sorted(c)
    assert calibrate_obf_c(5, alpha=0.10, n_sims=100000, seed=1) < calibrate_obf_c(5, alpha=0.05, n_sims=100000, seed=1)


def test_calibrate_reproducible():
    assert calibrate_obf_c(5, n_sims=1000, seed=3) == calibrate_obf_c(5, n_sims=1000, seed=3)
    assert calibrate_obf_c(5, n_sims=1000, seed=3) != calibrate_obf_c(5, n_sims=1000, seed=4)


@pytest.mark.parametrize("kwargs", [
    {"n_looks": 0}, {"n_looks": 2.5}, {"n_looks": 5, "alpha": 0},
    {"n_looks": 5, "alpha": 1}, {"n_looks": 5, "n_sims": 0},
])
def test_calibrate_invalid_raises(kwargs):
    with pytest.raises(ValueError):
        calibrate_obf_c(**kwargs)


# sequential_fp

@pytest.mark.parametrize("looks, c", [(10, 2.09), (5, 2.04)])
def test_sequential_controls_false_positives(looks, c):
    assert 0.04 <= sequential_fp(N_TOTAL, P, looks, c, N_SIMS, seed=1) <= 0.06


def test_sequential_with_calibrated_c_is_alpha():
    c = calibrate_obf_c(10, n_sims=200000, seed=7)
    assert 0.04 <= sequential_fp(N_TOTAL, P, 10, c, N_SIMS, seed=2) <= 0.06


def test_sequential_fixes_peeking():
    same_seed = dict(n_sims=N_SIMS, seed=1)
    assert sequential_fp(N_TOTAL, P, 10, 2.09, **same_seed) < peeking_fp(N_TOTAL, P, 10, **same_seed) / 2


def test_sequential_one_look_matches_ordinary_test():
    assert sequential_fp(N_TOTAL, P, 1, norm.ppf(0.975), N_SIMS, seed=1) == pytest.approx(
        false_positive_rate(N_TOTAL, P, N_SIMS, seed=1), abs=0.001)


def test_sequential_huge_c_never_rejects():
    assert sequential_fp(N_TOTAL, P, 5, 100.0, 2000, seed=1) == 0.0


def test_sequential_reproducible_and_returns_float():
    a = sequential_fp(N_TOTAL, P, 5, 2.04, 2000, seed=3)
    assert isinstance(a, float) and a == sequential_fp(N_TOTAL, P, 5, 2.04, 2000, seed=3)


@pytest.mark.parametrize("args", [
    (0, P, 5, 2.04, 100, 0),       # no users
    (100, P, 0, 2.04, 100, 0),     # no looks
    (5, P, 10, 2.04, 100, 0),      # more looks than users
    (100.5, P, 5, 2.04, 100, 0),   # non-integer
    (100, P, 5, 2.04, 0, 0),       # no simulations
    (100, 0.0, 5, 2.04, 100, 0),   # p_control = 0
    (100, 1.0, 5, 2.04, 100, 0),   # p_control = 1
    (100, P, 5, 0, 100, 0),        # c = 0
    (100, P, 5, -2.0, 100, 0),     # c < 0
])
def test_sequential_invalid_raises(args):
    with pytest.raises(ValueError):
        sequential_fp(*args)


# true_lift: power of the rules (A/B instead of A/A)

def test_true_lift_default_is_aa():
    assert peeking_fp(N_TOTAL, P, 5, 2000, seed=3) == peeking_fp(N_TOTAL, P, 5, 2000, seed=3, true_lift=0.0)
    assert sequential_fp(N_TOTAL, P, 5, 2.04, 2000, seed=3) == sequential_fp(N_TOTAL, P, 5, 2.04, 2000, seed=3, true_lift=0.0)


def test_true_lift_one_look_matches_planned_power():
    # one look at 14,749 per arm is the ordinary test: power 0.80 (spec 4.2)
    assert peeking_fp(14749, P, 1, N_SIMS, seed=1, true_lift=0.01) == pytest.approx(0.80, abs=0.015)
    assert sequential_fp(14749, P, 1, norm.ppf(0.975), N_SIMS, seed=1, true_lift=0.01) == pytest.approx(0.80, abs=0.015)


def test_true_lift_power_ordering():
    # a real effect is detected more often than a null one; the stricter sequential rule detects less than peeking
    kw = dict(n_sims=N_SIMS, seed=2)
    seq = sequential_fp(14749, P, 10, 2.09, true_lift=0.01, **kw)
    peek = peeking_fp(14749, P, 10, true_lift=0.01, **kw)
    assert seq > sequential_fp(14749, P, 10, 2.09, **kw)
    assert peek >= seq


@pytest.mark.parametrize("func_args", [
    (peeking_fp, (100, 0.99, 5, 10, 0)),
    (sequential_fp, (100, 0.99, 5, 2.04, 10, 0)),
])
def test_true_lift_out_of_range_raises(func_args):
    func, args = func_args
    with pytest.raises(ValueError):
        func(*args, true_lift=0.02)   # treatment rate 1.01
    with pytest.raises(ValueError):
        func(*args, true_lift=-0.99)  # treatment rate 0
