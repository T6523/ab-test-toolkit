import pytest

from abkit.sequential import peeking_fp
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
