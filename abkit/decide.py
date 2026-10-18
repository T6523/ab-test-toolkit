from enum import Enum


class Verdict(Enum):
    """Outcome of decide. str() gives "NAME: explanation"."""
    LAUNCH = "LAUNCH"
    HARMFUL = "HARMFUL"
    STAT_SIG_NOT_PRACTICAL = "STAT_SIG_NOT_PRACTICAL"
    NO_MEANINGFUL_EFFECT = "NO_MEANINGFUL_EFFECT"
    INCONCLUSIVE = "INCONCLUSIVE"

    def __str__(self):
        return f"{self.name}: {explain(self)}"


_EXPLANATIONS = {
    Verdict.LAUNCH: (
        "The whole confidence interval is at or above the practical "
        "threshold: launch."),
    Verdict.HARMFUL: (
        "The whole interval is below zero: the change likely hurts the "
        "metric. Do not launch."),
    Verdict.STAT_SIG_NOT_PRACTICAL: (
        "The effect is real but even its upper end is below the practical "
        "threshold: do not launch unless the change is nearly free."),
    Verdict.NO_MEANINGFUL_EFFECT: (
        "The interval includes zero and its upper end is below the "
        "practical threshold: a lift worth shipping is ruled out. Do not "
        "launch."),
    Verdict.INCONCLUSIVE: (
        "The interval spans the practical threshold: the data cannot "
        "separate a worthwhile lift from a smaller one. Collect more "
        "data."),
}


def decide(low, high, delta) -> Verdict:
    """Map lift CI and practical threshold to a Verdict. Ordered rules, first match wins.

    Lift CI only, guardrail metrics not considered.

    Args:
        low, high: CI ends on absolute lift, proportion, low <= high
        delta: smallest lift worth shipping, > 0

    Returns:
        Verdict: LAUNCH, HARMFUL, STAT_SIG_NOT_PRACTICAL, NO_MEANINGFUL_EFFECT or INCONCLUSIVE

    Raises:
        ValueError: low > high, delta <= 0
    """
    if low > high:
        raise ValueError(f"low must be <= high, got low={low}, high={high}")
    if delta <= 0:
        raise ValueError(f"delta must be > 0, got {delta}")

    if low >= delta:
        return Verdict.LAUNCH
    if high < 0:
        return Verdict.HARMFUL
    if high < delta and low > 0:
        return Verdict.STAT_SIG_NOT_PRACTICAL
    if high < delta:  # low <= 0
        return Verdict.NO_MEANINGFUL_EFFECT
    return Verdict.INCONCLUSIVE


def explain(verdict: Verdict) -> str:
    """One-sentence explanation of a verdict.

    Args:
        verdict: Verdict member

    Returns:
        str: explanation
    """
    return _EXPLANATIONS[verdict]