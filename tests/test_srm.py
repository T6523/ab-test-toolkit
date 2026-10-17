import numpy as np
import pytest

from abkit.srm import srm_check


@pytest.mark.parametrize("args, kwargs, chi2, p_value, flag", [
    ((50000, 50000), {}, 0.0, 1.0, False),
    ((50000, 51000), {}, 9.901, 0.001652, False),  # just above the 0.001 threshold
    ((50000, 52000), {}, 39.216, 3.8e-10, True),
    ((480, 520), {}, 1.6, 0.2059, False),
    ((300, 700), {"expected_ratio": 0.3}, 0.0, 1.0, False),  # matches a 30/70 plan
])
def test_srm_check(args, kwargs, chi2, p_value, flag):
    r = srm_check(*args, **kwargs)
    assert r["chi2"] == pytest.approx(chi2, abs=1e-3)
    assert r["p_value"] == pytest.approx(p_value, rel=1e-2, abs=1e-9)
    assert r["flag"] is flag


def test_threshold_is_a_parameter():
    # p = 0.001652: not flagged at 0.001, flagged at 0.01
    assert srm_check(50000, 51000)["flag"] is False
    assert srm_check(50000, 51000, threshold=0.01)["flag"] is True


def test_one_empty_arm_is_flagged():
    assert srm_check(100, 0)["flag"] is True


def test_returns_plain_python_types():
    r = srm_check(np.int64(50000), np.int64(52000))
    assert type(r["chi2"]) is float
    assert type(r["p_value"]) is float
    assert type(r["flag"]) is bool


@pytest.mark.parametrize("args, kwargs", [
    ((0, 0), {}),
    ((-5, 10), {}),
    ((10, -5), {}),
    ((10.5, 10), {}),
    (("a", 10), {}),
    ((100, 100), {"expected_ratio": 0}),
    ((100, 100), {"expected_ratio": 1}),
    ((100, 100), {"expected_ratio": 1.5}),
    ((100, 100), {"threshold": 0}),
    ((100, 100), {"threshold": 1}),
])
def test_srm_invalid_raises(args, kwargs):
    with pytest.raises(ValueError):
        srm_check(*args, **kwargs)
