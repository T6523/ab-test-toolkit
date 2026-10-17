from math import sqrt
from numbers import Integral
from scipy.stats import norm


def _validate_counts(x_control, n_control, x_treat, n_treat):
    """Raise ValueError naming the bad argument, accepts NumPy integers."""
    args = {"x_control": x_control, "n_control": n_control,
            "x_treat": x_treat, "n_treat": n_treat}
    for name, v in args.items():
        if not isinstance(v, Integral):
            raise ValueError(f"{name} must be an integer, got {v!r}")
    for name in ("n_control", "n_treat"):
        if args[name] <= 0:
            raise ValueError(f"{name} must be > 0, got {args[name]}")
    for x, n in (("x_control", "n_control"), ("x_treat", "n_treat")):
        if not 0 <= args[x] <= args[n]:
            raise ValueError(f"{x} must be between 0 and {n} ({args[n]}), got {args[x]}")


def ztest_two_prop(x_control, n_control, x_treat, n_treat) -> dict:
    """Test whether control and treatment conversion rates differ.

    Two-sided z-test for two proportions. The standard error uses the
    pooled rate, since the null hypothesis says both arms share one rate.

    Return a dict with lift (treatment - control, as a proportion), z and
    p_value. If the pooled rate is 0 or 1 the test is undefined, so
    z = 0.0 and p_value = 1.0.

    Raise ValueError if a count is not an integer, n <= 0, or x is not
    between 0 and n.
    """
    _validate_counts(x_control, n_control, x_treat, n_treat)
    pc = x_control / n_control
    pt = x_treat / n_treat
    lift = pt - pc
    p_pool = (x_control + x_treat) / (n_control + n_treat)
    if p_pool == 0 or p_pool == 1:
        z,p = 0.0, 1.0
    else:
        se = sqrt(p_pool * (1-p_pool) * (1/n_control + 1/n_treat))  # z_test pooled
        z = lift / se
        p = 2 * norm.sf(abs(z))

    return {'lift':lift,'z':z,'p_value':p}


def lift_ci(x_control, n_control, x_treat, n_treat, alpha=0.05) -> tuple:
    """Return the confidence interval (low, high) for the absolute lift.

    Wald interval, lift +/- z * se. The standard error is unpooled (each
    arm uses its own rate) because the interval does not assume the null.
    Can disagree with ztest_two_prop on borderline cases, which uses the
    pooled standard error.

    Raise ValueError for invalid counts, or if alpha is not in (0, 1).
    """
    _validate_counts(x_control, n_control, x_treat, n_treat)

    if alpha <= 0 or 1<= alpha:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    pc = x_control / n_control
    pt = x_treat / n_treat
    lift = pt - pc
    se = sqrt(pc * (1-pc) /n_control + pt*(1-pt) / n_treat) # not pooled
    z = norm.ppf(1 - alpha/2)
    return lift - z*se, lift + z*se  


def relative_lift_ci(x_control, n_control, x_treat, n_treat, alpha=0.05) -> tuple:
    """Return the confidence interval (low, high) for the relative lift.

    The absolute interval from lift_ci divided by the control rate, so
    0.10 means +10%. This is an approximation: it treats the control rate
    as fixed, so the true interval is somewhat wider.

    Raise ValueError for invalid counts, alpha outside (0, 1), or a
    control rate of zero.
    """
    low, high = lift_ci(x_control, n_control, x_treat, n_treat, alpha=alpha)
    if x_control == 0:
        raise ValueError("control is zero")
    pc = x_control / n_control
    return low / pc , high / pc

