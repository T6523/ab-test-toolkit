from abkit.analysis import _validate_counts
from abkit.sim import get_rng, _check_positive_int
import numpy as np


def compare(x_control, n_control, x_treat, n_treat, draws=100_000, seed=0, epsilon=0.0005) -> dict:
    """Bayesian comparison of two conversion rates. Beta-Binomial, posterior sampling, expected-loss decision.

    Prior is Beta(1, 1) (uniform) on each rate. Prior and epsilon are assumptions, not data.

    Args:
        x_control, x_treat: conversions, int, 0 <= x <= n
        n_control, n_treat: users, int, > 0
        draws: posterior samples per arm, int, > 0
        seed: int
        epsilon: ship if expected loss is below this, > 0 (0.0005 = 0.05 pp)

    Returns:
        dict:
            mean_control: posterior mean, (1 + x) / (2 + n), exact
            mean_treat: posterior mean, exact
            p_b_beats_a: P(treatment rate > control rate)
            expected_loss_choose_b: average rate given up per user if shipping treatment
            ship_b: expected_loss_choose_b < epsilon

    Raises:
        ValueError: invalid counts, draws or epsilon
    """
    _validate_counts(x_control, n_control, x_treat, n_treat)
    _check_positive_int("draws", draws)
    if not epsilon > 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")

    rng = get_rng(seed)
    # Beta prior is conjugate to the Binomial likelihood: prior Beta(1, 1) and x conversions in n users
    # give the posterior Beta(1 + x, 1 + n - x)
    p_control = rng.beta(1 + x_control, 1 + n_control - x_control, draws)
    p_treat = rng.beta(1 + x_treat, 1 + n_treat - x_treat, draws)
    # mean of Beta(a, b) is a / (a + b), so the posterior mean is (1 + x) / (2 + n), no sampling needed
    mean_control = (1 + x_control) / (2 + n_control)
    mean_treat = (1 + x_treat) / (2 + n_treat)
    # posterior probability that treatment is better = fraction of draws where it is (Monte Carlo estimate)
    p_b_beats_a = (p_treat > p_control).mean()
    # expected loss of choosing B = E[max(p_control - p_treat, 0)]: the rate given up on the draws
    # where control was actually better, averaged over all draws (0 where treatment wins)
    expected_loss_choose_b = np.maximum(p_control - p_treat, 0).mean()
    ship_b = expected_loss_choose_b < epsilon
    return {'mean_control': mean_control, 'mean_treat': mean_treat,
            'p_b_beats_a': float(p_b_beats_a), 'expected_loss_choose_b': float(expected_loss_choose_b),
            'ship_b': bool(ship_b)}
