import pytest

from abkit.analysis import lift_ci
from abkit.decide import Verdict, decide, explain

DELTA = 0.005


@pytest.mark.parametrize("low, high, expected", [
    (0.0062, 0.0138, Verdict.LAUNCH),
    (0.00041, 0.00159, Verdict.STAT_SIG_NOT_PRACTICAL),
    (-0.009, 0.029, Verdict.INCONCLUSIVE),
    (-0.00322, 0.00422, Verdict.NO_MEANINGFUL_EFFECT),
    (-0.012, -0.003, Verdict.HARMFUL),
    (0.003, 0.009, Verdict.INCONCLUSIVE),   # significant, lower end below delta
    (0.005, 0.009, Verdict.LAUNCH),         # boundary: low == delta
    (-0.002, 0.005, Verdict.INCONCLUSIVE),  # boundary: high == delta
])
def test_decide(low, high, expected):
    assert decide(low, high, DELTA) is expected


@pytest.mark.parametrize("low, high, expected", [
    (0.0, 0.004, Verdict.NO_MEANINGFUL_EFFECT),  # low == 0 is not significant
    (0.0, 0.0, Verdict.NO_MEANINGFUL_EFFECT),
    (-0.01, 0.0, Verdict.NO_MEANINGFUL_EFFECT),  # high == 0 is not HARMFUL (high < 0)
    (0.001, 0.004, Verdict.STAT_SIG_NOT_PRACTICAL),
])
def test_decide_zero_boundaries(low, high, expected):
    assert decide(low, high, DELTA) is expected


def test_decide_takes_lift_ci_output():
    # fixture A: CI [+0.15 pp, +1.85 pp] straddles delta = 0.5 pp
    assert decide(*lift_ci(1000, 10000, 1100, 10000), DELTA) is Verdict.INCONCLUSIVE


@pytest.mark.parametrize("args", [
    (0.01, 0.0, DELTA),   # low > high
    (0.0, 0.01, 0.0),     # delta = 0
    (0.0, 0.01, -0.005),  # delta < 0
])
def test_decide_invalid_raises(args):
    with pytest.raises(ValueError):
        decide(*args)


def test_verdict_values_equal_names():
    assert all(v.value == v.name for v in Verdict)


@pytest.mark.parametrize("verdict", list(Verdict))
def test_explain_returns_sentence(verdict):
    text = explain(verdict)
    assert isinstance(text, str) and text.endswith(".")


def test_str_shows_name_and_explanation():
    text = str(Verdict.LAUNCH)
    assert text.startswith("LAUNCH: ")
    assert explain(Verdict.LAUNCH) in text
