from math import ceil, sqrt
from scipy.stats import norm 
from scipy.optimize import brentq

def _validate_planning(p_control, alpha):
    """Raise ValueError if p_control, mde or alpha are out of range."""
    if not 0 < p_control < 1:
        raise ValueError(f"p_control must be between 0 and 1, got {p_control}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

def _validate_power(power):
    if not 0 < power < 1:
        raise ValueError(f"power must be between 0 and 1, got {power}")

def _validate_mde(p_control, mde):
    if mde == 0:
            raise ValueError("mde must be non-zero")
    if not 0 < p_control + mde < 1:
        raise ValueError(
            f"p_control + mde must be between 0 and 1, got {p_control + mde}")

def _n_float(p_control, mde, alpha, power) -> float:
    """Return the unrounded n per arm for a two-sided two-proportion z-test.

    By the CLT each arm's rate is approximately normal, so the difference
    D = p2_hat - p1_hat ~ N(mde, (p1(1-p1) + p2(1-p2)) / n).
    The test rejects when |D| / se > z_{1-alpha/2}, which under the
    alternative has power Phi(mde / se - z_{1-alpha/2}). Setting this equal
    to the target power and solving for n gives the formula below.
    """
    _validate_planning(p_control, alpha)
    _validate_power(power)
    _validate_mde(p_control, mde)
    
    p1= p_control
    p2 = p1 + mde
    var = (p1*(1-p1) + p2*(1-p2))
    n = (norm.ppf(1-alpha/2) + norm.ppf(power))**2 * var / mde ** 2
    return n

def sample_size_per_arm(p_control, mde, alpha=0.05, power=0.8, round_up=True) -> int:
    """Return the users needed in each arm to detect a lift of mde.

    Two-sided two-proportion z-test at the given alpha and power. Return an
    int rounded up if round_up, otherwise the unrounded float.

    Raise ValueError for invalid p_control, mde, alpha or power.
    """
    n = _n_float(p_control, mde, alpha, power)
    if round_up:
        n = ceil(n)
    return n

def power_at_n(p_control, mde, n_per_arm ,alpha=0.05) -> float:
    """Return the power of the test to detect a lift of mde with n_per_arm.

    Normal approximation, Phi(|mde| / se - z_{1-alpha/2}). The chance of
    rejecting in the wrong direction is negligible and ignored.

    Raise ValueError for invalid p_control, mde, alpha or n_per_arm.
    """
    _validate_planning(p_control, alpha)
    _validate_mde(p_control, mde)
    if not n_per_arm > 0:
        raise ValueError("n per arm must be positive")

    p1, p2 = p_control, p_control + mde
    se = sqrt((p1*(1-p1) + p2*(1-p2)) / n_per_arm)
    power = norm.cdf(abs(mde)/se - norm.ppf(1-alpha/2))
    return float(power)

def mde_at_n(p_control, n_per_arm ,alpha=0.05, power=0.8) -> float:
    _validate_planning(p_control, alpha)
    _validate_power(power)
    if not n_per_arm > 0:
        raise ValueError("n per arm must be positive")
    try:
        mde = brentq(lambda mde:_n_float(p_control,mde,alpha,power) - n_per_arm, 1e-9, 1-p_control-1e-9)
    except ValueError:
        raise ValueError(f"no mde found for n_per_arm={n_per_arm}, try larger n") from None
    return mde