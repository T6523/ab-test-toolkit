from abkit.sim import get_rng, analyze_batch, _check_positive_int
import numpy as np

def peeking_fp(n_total_per_arm, p_control, n_looks, n_sims, seed, alpha=0.05) -> float:
    """False-positive rate when peeking. Ordinary z-test at every look, stop at first significant.

    A/A data (true lift 0). Counts accumulate over looks, drawn as binomial increments.
    n_total_per_arm not divisible by n_looks -> effective total is n_looks * (n_total_per_arm // n_looks).

    Args:
        n_total_per_arm: users per arm at the final look, int, >= n_looks
        p_control: rate in both arms, in (0, 1)
        n_looks: equally spaced looks, int, > 0
        n_sims: number of simulated tests, int, > 0
        seed: int
        alpha: significance level at each look, in (0, 1)

    Returns:
        float: fraction of tests significant at any look

    Raises:
        ValueError: invalid n_total_per_arm, n_looks, n_sims, p_control or alpha
    """
    _check_positive_int("n_total_per_arm", n_total_per_arm)
    _check_positive_int("n_looks", n_looks)
    _check_positive_int("n_sims", n_sims)
    if n_looks > n_total_per_arm:
        raise ValueError(
            f"n_looks must be <= n_total_per_arm, got {n_looks} > {n_total_per_arm}")
    if not 0 < p_control < 1:
        raise ValueError(f"p_control must be between 0 and 1, got {p_control}")

    step = n_total_per_arm // n_looks
    rng = get_rng(seed)
    x_control, x_treat = np.zeros(n_sims, dtype=int), np.zeros(n_sims, dtype=int)
    hit = np.full(n_sims, False)
    n_per_arm_accum = step
    for k in range(1, n_looks + 1):
        # new users in each interval: Binomial(step, p), added to the running counts
        x_control += rng.binomial(step, p_control, n_sims)
        x_treat += rng.binomial(step, p_control, n_sims)
        # analyze_batch raises ValueError for an invalid alpha
        r = analyze_batch(x_control, x_treat, n_per_arm_accum, alpha)
        # once a test is significant at any look it stays a hit (it would have been stopped)
        hit |= r['significant']
        n_per_arm_accum += step
    return float(hit.mean())
