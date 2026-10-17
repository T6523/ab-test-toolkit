from numbers import Integral
from scipy.stats import chisquare


def srm_check(n_control, n_treat, expected_ratio=0.5, threshold=0.001) -> dict:
    """Check arm sizes against the planned split

    Return chi2, p_value and flag (p_value < threshold)

    Raise ValueError for invalid counts, expected_ratio or threshold
    """
    for name, n in (("n_control", n_control), ("n_treat", n_treat)):
        if not isinstance(n, Integral) or n < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {n}")
    if n_control + n_treat == 0:
        raise ValueError("n_control + n_treat must be > 0")
    if not 0 < expected_ratio < 1:
        raise ValueError(f"expected_ratio must be between 0 and 1, got {expected_ratio}")
    if not 0 < threshold < 1:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")

    total = n_control + n_treat
    expected_control = total * expected_ratio
    expected_treat = total - expected_control
    chi2, p = chisquare([n_control, n_treat], [expected_control, expected_treat])
    flag = p < threshold
    # plain Python types so the result support JSON
    return {"chi2": float(chi2), "p_value": float(p), "flag": bool(flag)}
