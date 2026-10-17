from math import sqrt
from numbers import Integral
from scipy.stats import norm


def _validate_counts(x_control, n_control, x_treat, n_treat):
    """Validate counts. Accepts NumPy integers.

    Args:
        x_control, x_treat: conversions
        n_control, n_treat: users

    Raises:
        ValueError: names bad argument; non-integer, n <= 0, x outside [0, n]
    """
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
    """Two-sided two-proportion z-test. Pooled SE (null: same rate).

    Args:
        x_control, x_treat: conversions, int, 0 <= x <= n
        n_control, n_treat: users, int, > 0

    Returns:
        dict:
            lift: treat - control, proportion
            z: test statistic
            p_value: two-sided
        pooled rate 0 or 1 -> z=0.0, p_value=1.0

    Raises:
        ValueError: bad count, n <= 0, x outside [0, n]
    """
    _validate_counts(x_control, n_control, x_treat, n_treat)
    pc = x_control / n_control
    pt = x_treat / n_treat
    lift = pt - pc
    p_pool = (x_control + x_treat) / (n_control + n_treat)
    if p_pool == 0 or p_pool == 1:
        z,p = 0.0, 1.0  # se would be 0, test undefined
    else:
        # H0: pc = pt, so both arms share p_pool and Var(lift) = p(1-p)(1/n_c + 1/n_t)
        # CLT: lift ~ N(0, se^2) under H0, so z = lift / se ~ N(0, 1)
        se = sqrt(p_pool * (1-p_pool) * (1/n_control + 1/n_treat))
        z = lift / se
        # two-sided: P(|Z| > |z|) = 2 * P(Z > |z|)
        p = 2 * norm.sf(abs(z))

    return {'lift':lift,'z':z,'p_value':p}


def lift_ci(x_control, n_control, x_treat, n_treat, alpha=0.05) -> tuple:
    """Wald CI for absolute lift. Unpooled SE (CI does not assume null).

    Can disagree with ztest_two_prop on borderline cases (pooled SE).

    Args:
        x_control, x_treat: conversions, int, 0 <= x <= n
        n_control, n_treat: users, int, > 0
        alpha: 1 - confidence level, in (0, 1)

    Returns:
        tuple: (low, high), absolute lift, proportion

    Raises:
        ValueError: bad count, alpha outside (0, 1)
    """
    _validate_counts(x_control, n_control, x_treat, n_treat)

    if alpha <= 0 or 1<= alpha:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    pc = x_control / n_control
    pt = x_treat / n_treat
    lift = pt - pc
    # no null here, so each arm keeps its own variance: Var(lift) = pc(1-pc)/n_c + pt(1-pt)/n_t
    se = sqrt(pc * (1-pc) /n_control + pt*(1-pt) / n_treat)
    # CLT: (lift - true lift) / se ~ N(0, 1), so lift +/- z_{1-alpha/2} * se covers true lift w.p. 1 - alpha
    z = norm.ppf(1 - alpha/2)
    return lift - z*se, lift + z*se  


def relative_lift_ci(x_control, n_control, x_treat, n_treat, alpha=0.05) -> tuple:
    """CI for relative lift: lift_ci / control rate.

    Approximate, control rate treated as fixed (true CI wider).

    Args:
        x_control, x_treat: conversions, int, 0 <= x <= n
        n_control, n_treat: users, int, > 0
        alpha: 1 - confidence level, in (0, 1)

    Returns:
        tuple: (low, high), relative lift, 0.10 = +10%

    Raises:
        ValueError: bad count, alpha outside (0, 1), control rate 0
    """
    low, high = lift_ci(x_control, n_control, x_treat, n_treat, alpha=alpha)
    if x_control == 0:
        raise ValueError("control is zero")
    pc = x_control / n_control
    # relative lift = absolute lift / pc, so divide both ends by pc (ignores noise in pc)
    return low / pc , high / pc

