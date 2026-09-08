"""Binomial scores for independent sampling comparisons.

Two sampled proportions both have uncertainty. Stim's sampled estimate is
not an exact reference probability. Keep the existing six-sigma threshold;
use the pooled null variance when comparing two independent shot samples.
"""
import math


def proportion_difference_sigma(p_a, shots_a, p_b, shots_b):
    """Absolute pooled two-sample score (also supports unequal sample sizes)."""
    if shots_a <= 0 or shots_b <= 0:
        raise ValueError("shot counts must be positive")
    if not 0 <= p_a <= 1 or not 0 <= p_b <= 1:
        raise ValueError("proportions must be finite and in [0, 1]")
    pooled = (p_a * shots_a + p_b * shots_b) / (shots_a + shots_b)
    variance = pooled * (1 - pooled) * (1 / shots_a + 1 / shots_b)
    if variance == 0:
        return 0.0 if p_a == p_b else math.inf
    return float(abs(p_a - p_b) / math.sqrt(variance))


def binomial_reference_sigma(observed, shots, expected):
    """Absolute score against an analytic probability; impossible events fail."""
    if shots <= 0:
        raise ValueError("shot count must be positive")
    if not 0 <= observed <= 1 or not 0 <= expected <= 1:
        raise ValueError("proportions must be finite and in [0, 1]")
    variance = expected * (1 - expected) / shots
    if variance == 0:
        return 0.0 if observed == expected else math.inf
    return float(abs(observed - expected) / math.sqrt(variance))
