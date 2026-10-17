import numpy as np
import pytest
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

from abkit.analysis import lift_ci, relative_lift_ci, ztest_two_prop

FIXTURE_A = (1000, 10000, 1100, 10000)


@pytest.mark.parametrize("args, lift, z, p_value", [
    (FIXTURE_A, 0.01, 2.306635, 0.021075),
    ((500, 5000, 500, 5000), 0.0, 0.0, 1.0),
    ((1200, 10000, 1000, 10000), -0.02, -4.519846, 6.19e-06),
    ((30, 300, 45, 300), 0.05, 1.851640, 0.064078),
    ((0, 100, 0, 100), 0.0, 0.0, 1.0),  # pooled rate 0: test undefined
    ((100, 100, 100, 100), 0.0, 0.0, 1.0),  # pooled rate 1: test undefined
])
def test_ztest_two_prop(args, lift, z, p_value):
    r = ztest_two_prop(*args)
    assert r["lift"] == pytest.approx(lift, abs=1e-9)
    assert r["z"] == pytest.approx(z, abs=1e-6)
    assert r["p_value"] == pytest.approx(p_value, abs=1e-6)


@pytest.mark.parametrize("args, kwargs, lo, hi", [
    (FIXTURE_A, {}, 0.001504, 0.018496),
    (FIXTURE_A, {"alpha": 0.01}, -0.001166, 0.021166),
    ((1200, 10000, 1000, 10000), {}, -0.028668, -0.011332),
    ((500, 5000, 500, 5000), {}, -0.011760, 0.011760),
])
def test_lift_ci(args, kwargs, lo, hi):
    assert lift_ci(*args, **kwargs) == pytest.approx((lo, hi), abs=1e-6)


@pytest.mark.parametrize("args, lo, hi", [
    (FIXTURE_A, 0.01504, 0.18496),
    ((1200, 10000, 1000, 10000), -0.23890, -0.09443),
])
def test_relative_lift_ci(args, lo, hi):
    assert relative_lift_ci(*args) == pytest.approx((lo, hi), abs=1e-5)


def test_ci_is_wider_at_higher_confidence():
    lo95, hi95 = lift_ci(*FIXTURE_A)
    lo99, hi99 = lift_ci(*FIXTURE_A, alpha=0.01)
    assert lo99 < lo95 and hi99 > hi95


def test_accepts_numpy_integers():
    args = [np.int64(v) for v in FIXTURE_A]
    assert ztest_two_prop(*args)["z"] == pytest.approx(2.306635, abs=1e-6)


@pytest.mark.parametrize("func", [ztest_two_prop, lift_ci, relative_lift_ci])
@pytest.mark.parametrize("args", [
    (10, 5, 1, 10),       # x_control > n_control
    (1, 10, 11, 10),      # x_treat > n_treat
    (-1, 10, 1, 10),      # negative count
    (1, 0, 1, 10),        # n_control = 0
    (1, 10, 1, -5),       # n_treat < 0
    (1.5, 10, 1, 10),     # non-integer count
    (1, "10", 1, 10),     # wrong type
])
def test_invalid_counts_raise(func, args):
    with pytest.raises(ValueError):
        func(*args)


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.5])
def test_invalid_alpha_raises(alpha):
    with pytest.raises(ValueError):
        lift_ci(*FIXTURE_A, alpha=alpha)
    with pytest.raises(ValueError):
        relative_lift_ci(*FIXTURE_A, alpha=alpha)


def test_relative_ci_zero_control_raises():
    with pytest.raises(ValueError):
        relative_lift_ci(0, 10000, 5, 10000)


# statsmodels cross-checks (fixture A). proportions_ztest pools the variance by
# default under the null (value=0), same as ours.
def test_p_value_matches_statsmodels():
    _, p = proportions_ztest([1100, 1000], [10000, 10000])
    assert ztest_two_prop(*FIXTURE_A)["p_value"] == pytest.approx(p, abs=1e-6)


def test_ci_matches_statsmodels():
    # statsmodels treats the first arm as the minuend, so passing treatment
    # first gives treatment - control, the same sign as ours.
    ref = confint_proportions_2indep(1100, 10000, 1000, 10000,
                                     method="wald", compare="diff")
    assert lift_ci(*FIXTURE_A) == pytest.approx(ref, abs=1e-9)
