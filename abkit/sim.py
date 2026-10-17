import warnings
from numbers import Integral

import numpy as np
from scipy.stats import norm


def get_rng(seed):
    return np.random.default_rng(seed=seed)


def _check_positive_int(name, value):
    if not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def simulate_ab(n_per_arm, p_control, true_lift, n_sims, seed) -> dict:
    """Simulate n_sims independent A/B tests with a known true lift.

    Each test has n_per_arm users per arm. Control converts at p_control and
    treatment at p_control + true_lift. Conversion counts are drawn directly
    from a binomial, which is equivalent to simulating users one by one.

    Return a dict with integer arrays x_control and x_treat, each of shape
    (n_sims,). The same seed gives the same arrays.

    Raise ValueError if n_per_arm or n_sims is not a positive integer, or if
    p_control or p_control + true_lift is not strictly between 0 and 1.
    """
    _check_positive_int("n_per_arm", n_per_arm)
    _check_positive_int("n_sims", n_sims)
    if not 0 < p_control < 1:
        raise ValueError(f"p_control must be between 0 and 1, got {p_control:.8f}")
    if not 0 < p_control + true_lift < 1:
        raise ValueError(
            f"p_control + true_lift must be between 0 and 1, got {p_control + true_lift:.8f}")

    rng = get_rng(seed)
    x_control = rng.binomial(n_per_arm, p_control, size=n_sims)
    x_treat = rng.binomial(n_per_arm, p_control + true_lift, size=n_sims)
    return {'x_control': x_control, 'x_treat': x_treat}  # shape (n_sims,)


def analyze_batch(x_control, x_treat, n_per_arm, alpha=0.05) -> dict:
    """Vectorized ztest_two_prop and lift_ci for many tests at once.

    x_control and x_treat are arrays of conversion counts with n_per_arm
    users in each arm. Give the same numbers as the scalar functions in
    analysis.py, element by element.

    Return a dict of arrays: lift (treatment - control), z and p_value from
    the pooled z-test, low and high from the unpooled Wald CI, and
    significant (p_value < alpha). Where the pooled rate is 0 or 1 the test
    is undefined, so z = 0 and p_value = 1.

    Raise ValueError for an invalid alpha or n_per_arm, counts that are not
    integers or lie outside 0..n_per_arm, or arrays of different shapes.
    """
    if alpha <= 0 or 1 <= alpha:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    _check_positive_int("n_per_arm", n_per_arm)

    x_control, x_treat = np.asarray(x_control), np.asarray(x_treat)

    if x_control.shape != x_treat.shape:
        raise ValueError(
            f"x_control and x_treat shape must be equal, got {x_control.shape}, {x_treat.shape}")
    for name, x in (("x_control", x_control), ("x_treat", x_treat)):
        if not np.issubdtype(x.dtype, np.integer):
            raise ValueError(f"{name} must contain integers, got dtype {x.dtype}")
        if not np.all((0 <= x) & (x <= n_per_arm)):
            raise ValueError(f"{name} must be between 0 and {n_per_arm}")

    pc, pt = x_control / n_per_arm, x_treat / n_per_arm
    lift = pt - pc

    # pooled, null assumes the same rate in both arms
    p_pool = (x_control + x_treat) / (2 * n_per_arm)
    se = np.sqrt(p_pool * (1 - p_pool) * (2 / n_per_arm))
    se = np.where(se != 0, se, 1)  # mask 0 as 1 to prevent dividing by 0
    # if p_pool is 0 or 1 set z=0.0, p=1.0
    ok = (p_pool > 0) & (p_pool < 1)
    z = np.where(ok, lift / se, 0.0)
    p = np.where(ok, 2 * norm.sf(abs(z)), 1.0)
    sig = p < alpha

    # low, high do not assume the null
    se_unpooled = np.sqrt((pc * (1 - pc) + pt * (1 - pt)) / n_per_arm)
    z_unpooled = norm.ppf(1 - alpha / 2)
    low = lift - z_unpooled * se_unpooled
    high = lift + z_unpooled * se_unpooled

    return {'lift': lift, 'z': z, 'p_value': p, 'low': low, 'high': high, 'significant': sig}


def false_positive_rate(n_per_arm, p_control, n_sims, seed, alpha=0.05) -> float:
    """Return the fraction of A/A tests (true_lift = 0) that are significant.

    A correct test gives about alpha.
    """
    simul = simulate_ab(n_per_arm, p_control, 0, n_sims, seed)
    A_A = analyze_batch(simul['x_control'], simul['x_treat'], n_per_arm, alpha)
    return float(A_A['significant'].mean())


def ci_coverage(n_per_arm, p_control, true_lift, n_sims, seed, alpha=0.05) -> float:
    """Return the fraction of simulated CIs that contain the true lift.

    A correct (1 - alpha) CI covers about 1 - alpha.
    """
    simul = simulate_ab(n_per_arm, p_control, true_lift, n_sims, seed)
    A_B = analyze_batch(simul['x_control'], simul['x_treat'], n_per_arm, alpha)
    return float(((A_B['low'] <= true_lift) & (true_lift <= A_B['high'])).mean())


def _power(n_per_arm, p_control, true_lift, n_sims, seed, alpha=0.05):
    simul = simulate_ab(n_per_arm, p_control, true_lift, n_sims, seed)
    A_B = analyze_batch(simul['x_control'], simul['x_treat'], n_per_arm, alpha)
    return float(A_B['significant'].mean())


def power_curve(p_control, true_lifts, n_grid, n_sims, seed, alpha=0.05) -> dict:
    """Return {true_lift: [simulated power at each n in n_grid]}.

    Power is the fraction of simulated tests that are significant. Each
    sample size in n_grid uses its own derived seed (seed + its index).
    """
    return {true_lift: [_power(n, p_control, true_lift, n_sims, seed + i, alpha)
                        for i, n in enumerate(n_grid)]
            for true_lift in true_lifts}


def exaggeration_stats(n_per_arm, p_control, true_lift, n_sims, seed, alpha=0.05) -> dict:
    """Measure how much significant results overstate the true lift.

    For the significant simulations return:
      exaggeration_ratio: mean(|estimated lift|) / |true_lift| (type M error)
      type_s: fraction whose estimate has the wrong sign (type S error)
    plus power and n_significant (the number of significant runs).

    If no run is significant, exaggeration_ratio and type_s are nan and a
    warning is issued.

    Raise ValueError if true_lift is 0 (the ratio is undefined) or for
    invalid simulation arguments.
    """
    if true_lift == 0:
        raise ValueError('true_lift must not be 0 for the ratio to be defined')

    simul = simulate_ab(n_per_arm, p_control, true_lift, n_sims, seed)
    A_B = analyze_batch(simul['x_control'], simul['x_treat'], n_per_arm, alpha)
    sig = A_B['significant']
    n_significant = int(sig.sum())
    power = float(sig.mean())
    if n_significant == 0:
        warnings.warn("no significant runs; exaggeration_ratio and type_s are undefined")
        exaggeration_ratio, type_s = float('nan'), float('nan')
    else:
        lift_sig = A_B['lift'][sig]
        exaggeration_ratio = float(abs(lift_sig).mean() / abs(true_lift))
        type_s = float((np.sign(lift_sig) != np.sign(true_lift)).mean())  # wrong sign
    return {'power': power, 'n_significant': n_significant,
            'exaggeration_ratio': exaggeration_ratio, 'type_s': type_s}
