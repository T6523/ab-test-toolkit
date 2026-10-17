import warnings
from numbers import Integral

import numpy as np
from scipy.stats import norm


def get_rng(seed):
    """Seeded NumPy random generator, no global state.

    Args:
        seed: int

    Returns:
        numpy.random.Generator
    """
    return np.random.default_rng(seed=seed)


def _check_positive_int(name, value):
    """Validate positive integer.

    Args:
        name: argument name for the error message
        value: value to check

    Raises:
        ValueError: not an integer, or < 1
    """
    if not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def simulate_ab(n_per_arm, p_control, true_lift, n_sims, seed) -> dict:
    """Simulate n_sims A/B tests with known true lift. Binomial draws per arm.

    Args:
        n_per_arm: users per arm, int, > 0
        p_control: control rate, in (0, 1)
        true_lift: treatment rate - control rate, p_control + true_lift in (0, 1)
        n_sims: number of tests, int, > 0
        seed: int, same seed -> same arrays

    Returns:
        dict:
            x_control: conversions, int array, shape (n_sims,)
            x_treat: conversions, int array, shape (n_sims,)

    Raises:
        ValueError: n_per_arm or n_sims not positive int, rate outside (0, 1)
    """
    _check_positive_int("n_per_arm", n_per_arm)
    _check_positive_int("n_sims", n_sims)
    if not 0 < p_control < 1:
        raise ValueError(f"p_control must be between 0 and 1, got {p_control:.8f}")
    if not 0 < p_control + true_lift < 1:
        raise ValueError(
            f"p_control + true_lift must be between 0 and 1, got {p_control + true_lift:.8f}")

    rng = get_rng(seed)
    # sum of n_per_arm Bernoulli(p) users ~ Binomial(n_per_arm, p), so draw counts directly
    x_control = rng.binomial(n_per_arm, p_control, size=n_sims)
    x_treat = rng.binomial(n_per_arm, p_control + true_lift, size=n_sims)
    return {'x_control': x_control, 'x_treat': x_treat}  # shape (n_sims,)


def analyze_batch(x_control, x_treat, n_per_arm, alpha=0.05) -> dict:
    """Vectorized ztest_two_prop and lift_ci. Same numbers as scalar, element by element.

    Args:
        x_control, x_treat: conversions, int arrays, same shape, 0 <= x <= n_per_arm
        n_per_arm: users per arm, int, > 0
        alpha: significance level, in (0, 1)

    Returns:
        dict (arrays, same shape as x_control):
            lift: treat - control, proportion
            z: pooled z-test statistic
            p_value: two-sided
            low, high: unpooled Wald CI
            significant: p_value < alpha
        pooled rate 0 or 1 -> z=0, p_value=1

    Raises:
        ValueError: invalid alpha or n_per_arm, non-integer or out-of-range counts, shape mismatch
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
    """Type I error rate by simulation. A/A tests (true_lift = 0), correct test gives about alpha.

    Args:
        n_per_arm: users per arm, int, > 0
        p_control: shared rate, in (0, 1)
        n_sims: number of tests, int, > 0
        seed: int
        alpha: significance level, in (0, 1)

    Returns:
        float: fraction significant

    Raises:
        ValueError: see simulate_ab, analyze_batch
    """
    simul = simulate_ab(n_per_arm, p_control, 0, n_sims, seed)
    A_A = analyze_batch(simul['x_control'], simul['x_treat'], n_per_arm, alpha)
    return float(A_A['significant'].mean())


def ci_coverage(n_per_arm, p_control, true_lift, n_sims, seed, alpha=0.05) -> float:
    """CI coverage by simulation. Correct (1 - alpha) CI covers about 1 - alpha.

    Args:
        n_per_arm: users per arm, int, > 0
        p_control: control rate, in (0, 1)
        true_lift: true absolute lift, p_control + true_lift in (0, 1)
        n_sims: number of tests, int, > 0
        seed: int
        alpha: 1 - confidence level, in (0, 1)

    Returns:
        float: fraction of CIs containing true_lift

    Raises:
        ValueError: see simulate_ab, analyze_batch
    """
    simul = simulate_ab(n_per_arm, p_control, true_lift, n_sims, seed)
    A_B = analyze_batch(simul['x_control'], simul['x_treat'], n_per_arm, alpha)
    return float(((A_B['low'] <= true_lift) & (true_lift <= A_B['high'])).mean())


def _power(n_per_arm, p_control, true_lift, n_sims, seed, alpha=0.05):
    """Simulated power at one n. Same arguments as ci_coverage.

    Returns:
        float: fraction significant
    """
    simul = simulate_ab(n_per_arm, p_control, true_lift, n_sims, seed)
    A_B = analyze_batch(simul['x_control'], simul['x_treat'], n_per_arm, alpha)
    return float(A_B['significant'].mean())


def power_curve(p_control, true_lifts, n_grid, n_sims, seed, alpha=0.05) -> dict:
    """Simulated power for each true lift and sample size. Seed + index per n.

    Args:
        p_control: control rate, in (0, 1)
        true_lifts: iterable of true absolute lifts
        n_grid: iterable of n per arm
        n_sims: number of tests per cell, int, > 0
        seed: int, n_grid[i] uses seed + i
        alpha: significance level, in (0, 1)

    Returns:
        dict:
            true_lift: list of power, one per n in n_grid

    Raises:
        ValueError: see simulate_ab, analyze_batch
    """
    return {true_lift: [_power(n, p_control, true_lift, n_sims, seed + i, alpha)
                        for i, n in enumerate(n_grid)]
            for true_lift in true_lifts}


def exaggeration_stats(n_per_arm, p_control, true_lift, n_sims, seed, alpha=0.05) -> dict:
    """Type M (exaggeration) and type S (sign) errors among significant runs (Gelman and Carlin).

    Args:
        n_per_arm: users per arm, int, > 0
        p_control: control rate, in (0, 1)
        true_lift: true absolute lift, != 0, p_control + true_lift in (0, 1)
        n_sims: number of tests, int, > 0
        seed: int
        alpha: significance level, in (0, 1)

    Returns:
        dict:
            power: fraction significant
            n_significant: number of significant runs
            exaggeration_ratio: mean(|estimated lift|) / |true_lift|, significant runs
            type_s: fraction of significant runs with wrong sign
        none significant -> ratio and type_s are nan, warns

    Raises:
        ValueError: true_lift is 0, see simulate_ab, analyze_batch
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
