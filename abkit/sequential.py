from abkit.sim import get_rng, analyze_batch, _check_positive_int
import numpy as np
from math import sqrt

def peeking_fp(n_total_per_arm, p_control, n_looks, n_sims, seed, alpha=0.05, true_lift=0.0) -> float:
    """False-positive rate when peeking. Ordinary z-test at every look, stop at first significant.

    A/A data by default (true_lift = 0). Counts accumulate over looks, drawn as binomial increments.
    With true_lift != 0 the same rule gives the power of peeking instead.
    n_total_per_arm not divisible by n_looks -> effective total is n_looks * (n_total_per_arm // n_looks).

    Args:
        n_total_per_arm: users per arm at the final look, int, >= n_looks
        p_control: control rate, in (0, 1)
        n_looks: equally spaced looks, int, > 0
        n_sims: number of simulated tests, int, > 0
        seed: int
        alpha: significance level at each look, in (0, 1)
        true_lift: treatment rate - control rate, p_control + true_lift in (0, 1)

    Returns:
        float: fraction of tests significant at any look (false-positive rate if true_lift = 0, else power)

    Raises:
        ValueError: invalid n_total_per_arm, n_looks, n_sims, p_control, true_lift or alpha
    """
    _check_positive_int("n_total_per_arm", n_total_per_arm)
    _check_positive_int("n_looks", n_looks)
    _check_positive_int("n_sims", n_sims)
    if n_looks > n_total_per_arm:
        raise ValueError(
            f"n_looks must be <= n_total_per_arm, got {n_looks} > {n_total_per_arm}")
    if not 0 < p_control < 1:
        raise ValueError(f"p_control must be between 0 and 1, got {p_control}")
    if not 0 < p_control + true_lift < 1:
        raise ValueError(
            f"p_control + true_lift must be between 0 and 1, got {p_control + true_lift:.8f}")

    step = n_total_per_arm // n_looks
    rng = get_rng(seed)
    x_control, x_treat = np.zeros(n_sims, dtype=int), np.zeros(n_sims, dtype=int)
    hit = np.full(n_sims, False)
    n_per_arm_accum = step
    for _ in range(1, n_looks + 1):
        # new users in each interval: Binomial(step, p), added to the running counts
        x_control += rng.binomial(step, p_control, n_sims)
        x_treat += rng.binomial(step, p_control + true_lift, n_sims)
        # analyze_batch raises ValueError for an invalid alpha
        r = analyze_batch(x_control, x_treat, n_per_arm_accum, alpha)
        # once a test is significant at any look it stays a hit (it would have been stopped)
        hit |= r['significant']
        n_per_arm_accum += step
    return float(hit.mean())


def obf_boundaries(n_looks, c) -> list:
    """Critical z at each look, O'Brien-Fleming-type: c * sqrt(n_looks / k).

    Early looks demand a very large z, last look equals c.
    Same as a flat cutoff c * sqrt(n_looks) on the running sum of increments.

    Args:
        n_looks: number of equally spaced looks, int, > 0
        c: boundary constant, > 0 (from calibrate_obf_c)

    Returns:
        list: critical z for k = 1..n_looks

    Raises:
        ValueError: invalid n_looks or c
    """
    _check_positive_int("n_looks", n_looks)
    if not c > 0:
        raise ValueError(f"c must be > 0, got {c}")
    # O'Brien-Fleming shape: the cutoff shrinks like 1 / sqrt(k) as information k / K accumulates
    return [c * sqrt(n_looks / k) for k in  range(1,n_looks+1)]


def calibrate_obf_c(n_looks, alpha=0.05, n_sims=200000, seed=0) -> float:
    """Constant c so the boundary c * sqrt(n_looks / k) has overall false-positive rate alpha.

    Monte Carlo quantile of the maximum of a null Gaussian random walk (CLT approximation).

    Args:
        n_looks: number of equally spaced looks K, int, > 0
        alpha: overall significance level, in (0, 1)
        n_sims: number of simulated walks, int, > 0
        seed: int

    Returns:
        float: c

    Raises:
        ValueError: invalid n_looks, alpha or n_sims
    """
    _check_positive_int("n_looks", n_looks)
    _check_positive_int("n_sims", n_sims)
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

    rng = get_rng(seed)
    # H0: by the CLT each interval adds an independent N(0, 1) increment to the standardized
    # statistic, so the running sum S_k is a random walk and Z_k = S_k / sqrt(k) ~ N(0, 1) at every look
    norm = rng.normal(0,1, (n_sims, n_looks))
    s = np.cumsum(norm, axis=1)
    # reject at look k when |Z_k| >= c * sqrt(K / k), i.e. |Z_k| * sqrt(k / K) >= c
    # |Z_k| * sqrt(k / K) = |S_k| / sqrt(k) * sqrt(k) / sqrt(K) = |S_k| / sqrt(K), the sqrt(k) cancels
    # so any look rejects <=> M = max_k |S_k| / sqrt(K) >= c
    m = np.abs(s).max(axis=1) / sqrt(n_looks)
    # P(M >= c) = alpha, so c is the (1 - alpha) quantile of M
    c = np.quantile(m, 1- alpha)
    return float(c)


def sequential_fp(n_total_per_arm, p_control, n_looks, c, n_sims, seed, true_lift=0.0) -> float:
    """False-positive rate with calibrated sequential boundaries. Like peeking_fp, but significant if |z| >= c * sqrt(n_looks / k).

    A/A data by default (true_lift = 0). With true_lift != 0 it gives the power of the sequential rule.
    Valid only for the pre-planned number of equally spaced looks.
    n_total_per_arm not divisible by n_looks -> effective total is n_looks * (n_total_per_arm // n_looks).

    Args:
        n_total_per_arm: users per arm at the final look, int, >= n_looks
        p_control: control rate, in (0, 1)
        n_looks: equally spaced looks, int, > 0
        c: boundary constant (from calibrate_obf_c)
        n_sims: number of simulated tests, int, > 0
        seed: int
        true_lift: treatment rate - control rate, p_control + true_lift in (0, 1)

    Returns:
        float: fraction of tests with |z| >= boundary at any look (false-positive rate if true_lift = 0, else power)

    Raises:
        ValueError: invalid inputs
    """
    _check_positive_int("n_total_per_arm", n_total_per_arm)
    _check_positive_int("n_looks", n_looks)
    _check_positive_int("n_sims", n_sims)
    if n_looks > n_total_per_arm:
        raise ValueError(
            f"n_looks must be <= n_total_per_arm, got {n_looks} > {n_total_per_arm}")
    if not 0 < p_control < 1:
        raise ValueError(f"p_control must be between 0 and 1, got {p_control}")
    if not 0 < p_control + true_lift < 1:
        raise ValueError(
            f"p_control + true_lift must be between 0 and 1, got {p_control + true_lift:.8f}")
    if not c > 0:
        raise ValueError(f"c must be > 0, got {c}")

    step = n_total_per_arm // n_looks
    rng = get_rng(seed)
    x_control, x_treat = np.zeros(n_sims, dtype=int), np.zeros(n_sims, dtype=int)
    hit = np.full(n_sims, False)
    n_per_arm_accum = step
    # critical |z| per look; c came from calibrate_obf_c so that the chance of ANY look crossing is alpha
    boundary = obf_boundaries(n_looks, c)
    for k in range(1, n_looks + 1):
        # new users in each interval: Binomial(step, p), added to the running counts
        x_control += rng.binomial(step, p_control, n_sims)
        x_treat += rng.binomial(step, p_control + true_lift, n_sims)
        # analyze_batch raises ValueError for an invalid alpha
        r = analyze_batch(x_control, x_treat, n_per_arm_accum, 0.05)    # alpha doesn't matter for z
        # pooled z-test at look k, compared with the look's own (larger, early) cutoff instead of 1.96
        # once a test crosses at any look it stays a hit (it would have been stopped)
        hit |= np.abs(r['z']) >= boundary[k-1]
        n_per_arm_accum += step
    return float(hit.mean())
