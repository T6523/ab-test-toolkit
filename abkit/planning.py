from math import ceil, sqrt
from scipy.stats import norm 
from scipy.optimize import brentq

def _validate_planning(p_control, alpha):
    """Validate p_control and alpha.

    Raises:
        ValueError: p_control or alpha outside (0, 1)
    """
    if not 0 < p_control < 1:
        raise ValueError(f"p_control must be between 0 and 1, got {p_control}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

def _validate_power(power):
    """Validate power.

    Raises:
        ValueError: power outside (0, 1)
    """
    if not 0 < power < 1:
        raise ValueError(f"power must be between 0 and 1, got {power}")

def _validate_mde(p_control, mde):
    """Validate mde against p_control.

    Raises:
        ValueError: mde is 0, or p_control + mde outside (0, 1)
    """
    if mde == 0:
            raise ValueError("mde must be non-zero")
    if not 0 < p_control + mde < 1:
        raise ValueError(
            f"p_control + mde must be between 0 and 1, got {p_control + mde}")

def _n_float(p_control, mde, alpha, power) -> float:
    """Unrounded n per arm. CLT + power of two-sided two-proportion z-test, solved for n.

    Args:
        p_control: control rate, in (0, 1)
        mde: absolute lift to detect, != 0, p_control + mde in (0, 1)
        alpha: significance level, in (0, 1)
        power: target power, in (0, 1)

    Returns:
        float: n per arm

    Raises:
        ValueError: invalid p_control, mde, alpha or power
    """
    _validate_planning(p_control, alpha)
    _validate_power(power)
    _validate_mde(p_control, mde)

    p1= p_control
    p2 = p1 + mde
    var = (p1*(1-p1) + p2*(1-p2))
    # CLT: D = p2_hat - p1_hat ~ N(mde, var / n), se = sqrt(var / n)
    # reject when |D| / se > z_{1-alpha/2}, so under H1: power = Phi(mde / se - z_{1-alpha/2})
    # set power = Phi(z_power) -> mde / se = z_{1-alpha/2} + z_power
    # square both sides, se^2 = var / n -> n = (z_{1-alpha/2} + z_power)^2 * var / mde^2
    n = (norm.ppf(1-alpha/2) + norm.ppf(power))**2 * var / mde ** 2
    return n

def sample_size_per_arm(p_control, mde, alpha=0.05, power=0.8, round_up=True) -> int:
    """Users per arm to detect mde. Two-sided two-proportion z-test (see _n_float).

    Args:
        p_control: control rate, in (0, 1)
        mde: absolute lift to detect, != 0, p_control + mde in (0, 1)
        alpha: significance level, in (0, 1)
        power: target power, in (0, 1)
        round_up: ceil to int, else raw float

    Returns:
        int: n per arm (float if round_up=False)

    Raises:
        ValueError: invalid p_control, mde, alpha or power
    """
    n = _n_float(p_control, mde, alpha, power)
    if round_up:
        n = ceil(n)
    return n

def power_at_n(p_control, mde, n_per_arm ,alpha=0.05) -> float:
    """Power to detect mde at n_per_arm. CLT, normal approximation.

    Args:
        p_control: control rate, in (0, 1)
        mde: true absolute lift, != 0, p_control + mde in (0, 1)
        n_per_arm: users per arm, > 0
        alpha: significance level, in (0, 1)

    Returns:
        float: power, in (0, 1)

    Raises:
        ValueError: invalid p_control, mde, alpha or n_per_arm
    """
    _validate_planning(p_control, alpha)
    _validate_mde(p_control, mde)
    if not n_per_arm > 0:
        raise ValueError("n per arm must be positive")

    p1, p2 = p_control, p_control + mde
    se = sqrt((p1*(1-p1) + p2*(1-p2)) / n_per_arm)
    # CLT: D ~ N(mde, se^2), reject when |D| / se > z_{1-alpha/2}
    # power = P(reject | H1) = Phi(|mde| / se - z_{1-alpha/2}); wrong-direction rejection tail ignored
    power = norm.cdf(abs(mde)/se - norm.ppf(1-alpha/2))
    return float(power)

def mde_at_n(p_control, n_per_arm ,alpha=0.05, power=0.8) -> float:
    """Smallest detectable lift at n_per_arm. Root-find on _n_float (no closed form).

    Args:
        p_control: control rate, in (0, 1)
        n_per_arm: users per arm, > 0
        alpha: significance level, in (0, 1)
        power: target power, in (0, 1)

    Returns:
        float: mde, absolute lift, positive

    Raises:
        ValueError: invalid p_control, n_per_arm, alpha or power; no mde found
    """
    _validate_planning(p_control, alpha)
    _validate_power(power)
    if not n_per_arm > 0:
        raise ValueError("n per arm must be positive")
    # var depends on mde too, so n = f(mde) cannot be inverted; solve _n_float(mde) = n_per_arm
    # _n_float decreases in mde, brentq needs a sign change over the bracket
    try:
        mde = brentq(lambda mde:_n_float(p_control,mde,alpha,power) - n_per_arm, 1e-9, 1-p_control-1e-9)
    except ValueError:
        raise ValueError(f"no mde found for n_per_arm={n_per_arm}, try larger n") from None
    return mde