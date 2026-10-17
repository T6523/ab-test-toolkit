from enum import Enum


class Verdict(Enum):
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
        "metric; do not launch."),
    Verdict.STAT_SIG_NOT_PRACTICAL: (
        "The effect is real but even its upper end is below the practical "
        "threshold: do not launch unless the change is nearly free."),
    Verdict.NO_MEANINGFUL_EFFECT: (
        "The interval includes zero and its upper end is below the "
        "practical threshold: a lift worth shipping is ruled out; do not "
        "launch."),
    Verdict.INCONCLUSIVE: (
        "The interval spans the practical threshold: the data cannot "
        "separate a worthwhile lift from a smaller one; collect more "
        "data."),
}


def decide(low, high, delta) -> Verdict:
    """Turn a CI on absolute lift and a practical threshold into a verdict.

    low, high are the ends of the confidence interval on the absolute lift (a
    proportion) and delta is the smallest lift worth shipping. Rules are
    checked in order and the first match wins.

    Only the lift CI is considered; guardrail metrics are not looked at.

    Raise ValueError if low > high or delta <= 0.
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
    """Return a one-sentence explanation of a verdict."""
    return _EXPLANATIONS[verdict]