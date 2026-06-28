from __future__ import annotations

import math
import random
from statistics import mean
from typing import Iterable, List, Tuple


def paired_bootstrap_ci(
    baseline_scores: Iterable[float],
    candidate_scores: Iterable[float],
    iterations: int = 10000,
    confidence: float = 0.95,
    seed: int = 7,
) -> Tuple[float, float, float]:
    """Return mean paired delta and bootstrap confidence interval.

    Scores must be paired by benchmark item. The returned delta is candidate - baseline.
    """
    baseline = list(baseline_scores)
    candidate = list(candidate_scores)
    if len(baseline) != len(candidate):
        raise ValueError("Paired bootstrap requires equal-length score lists.")
    if not baseline:
        raise ValueError("At least one paired score is required.")

    rng = random.Random(seed)
    deltas: List[float] = []
    n = len(baseline)
    paired = list(zip(baseline, candidate))
    observed = mean(c - b for b, c in paired)

    for _ in range(iterations):
        sample = [paired[rng.randrange(n)] for _ in range(n)]
        deltas.append(mean(c - b for b, c in sample))

    deltas.sort()
    alpha = 1.0 - confidence
    lower_index = max(0, math.floor((alpha / 2) * iterations))
    upper_index = min(iterations - 1, math.ceil((1 - alpha / 2) * iterations) - 1)
    return observed, deltas[lower_index], deltas[upper_index]


def relative_lift(baseline: float, candidate: float) -> float:
    if baseline == 0:
        return 0.0
    return (candidate - baseline) / baseline


def is_practically_significant(
    delta: float,
    ci_lower: float,
    minimum_effect_size: float,
) -> bool:
    """Require both statistical and practical significance."""
    return ci_lower > 0 and delta >= minimum_effect_size

