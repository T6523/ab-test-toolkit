import pytest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from abkit.planning import mde_at_n, power_at_n, sample_size_per_arm


@pytest.mark.parametrize("p, mde, kwargs, n", [
    (0.10, 0.01, {}, 14749),
    (0.10, 0.005, {}, 57760),
    (0.10, 0.001, {}, 1419070),
    (0.05, 0.01, {}, 8155),
    (0.10, 0.01, {"power": 0.90}, 19744),
    (0.10, 0.01, {"alpha": 0.01}, 21945),
])
def test_sample_size_per_arm(p, mde, kwargs, n):
    result = sample_size_per_arm(p, mde, **kwargs)
    assert result == n
    assert isinstance(result, int)


def test_sample_size_float_form():
    assert sample_size_per_arm(0.10, 0.01, round_up=False) == pytest.approx(14748.045, abs=1e-3)


def test_sample_size_close_to_statsmodels():
    # statsmodels uses an arcsine (Cohen's h) approximation; observed gap is
    # about 0.03% (14744 vs 14749), well inside the 2% tolerance.
    ref = NormalIndPower().solve_power(
        effect_size=proportion_effectsize(0.11, 0.10), alpha=0.05, power=0.8, ratio=1)
    assert sample_size_per_arm(0.10, 0.01) == pytest.approx(ref, rel=0.02)


@pytest.mark.parametrize("p, mde, n, power", [
    (0.10, 0.01, 14749, 0.8000),
    (0.10, 0.01, 2000, 0.1766),
    (0.10, 0.01, 50000, 0.9993),
    (0.10, 0.005, 2000, 0.0751),
])
def test_power_at_n(p, mde, n, power):
    assert power_at_n(p, mde, n) == pytest.approx(power, abs=1e-4)


def test_power_increases_with_n():
    assert power_at_n(0.10, 0.01, 1000) < power_at_n(0.10, 0.01, 5000)


@pytest.mark.parametrize("p, n, mde", [
    (0.10, 14749, 0.009999669),
    (0.10, 2000, 0.028136),
    (0.10, 50000, 0.005378),
])
def test_mde_at_n(p, n, mde):
    assert mde_at_n(p, n) == pytest.approx(mde, abs=1e-6)


@pytest.mark.parametrize("m", [0.005, 0.01, 0.02])
def test_mde_round_trip(m):
    n = sample_size_per_arm(0.10, m, round_up=False)
    assert mde_at_n(0.10, n) == pytest.approx(m, rel=1e-6)


@pytest.mark.parametrize("kwargs", [
    {"p_control": 0.0, "mde": 0.01},
    {"p_control": 1.0, "mde": -0.01},
    {"p_control": 0.10, "mde": 0.0},
    {"p_control": 0.99, "mde": 0.02},  # p_control + mde > 1
    {"p_control": 0.10, "mde": 0.01, "alpha": 0},
    {"p_control": 0.10, "mde": 0.01, "alpha": 1},
    {"p_control": 0.10, "mde": 0.01, "power": 0},
    {"p_control": 0.10, "mde": 0.01, "power": 1},
])
def test_sample_size_invalid_raises(kwargs):
    with pytest.raises(ValueError):
        sample_size_per_arm(**kwargs)


@pytest.mark.parametrize("args", [
    (0.10, 0.0, 100),      # mde = 0
    (1.2, 0.01, 100),      # bad p_control
    (0.10, 0.01, 0),       # n = 0
    (0.10, 0.01, -5),      # n < 0
])
def test_power_at_n_invalid_raises(args):
    with pytest.raises(ValueError):
        power_at_n(*args)


@pytest.mark.parametrize("args", [
    (1.2, 100),            # bad p_control
    (0.10, 0),             # n = 0
    (0.10, -5),            # n < 0
    (0.95, 100),           # no mde below 0.05 needs only 100 users
])
def test_mde_at_n_invalid_raises(args):
    with pytest.raises(ValueError):
        mde_at_n(*args)


@pytest.mark.parametrize("kwargs", [{"alpha": 0}, {"power": 1.5}])
def test_mde_at_n_invalid_params_raise(kwargs):
    with pytest.raises(ValueError):
        mde_at_n(0.10, 2000, **kwargs)
