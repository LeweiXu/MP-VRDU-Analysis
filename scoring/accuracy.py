"""Document-level accuracy with bootstrap confidence intervals."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from config import BOOTSTRAP_CI_HIGH, BOOTSTRAP_CI_LOW, BOOTSTRAP_SEED, N_BOOTSTRAP


class AccuracyRow(Protocol):
    """Minimal row surface required for accuracy aggregation."""

    doc_id: str
    correct: bool


@dataclass(frozen=True)
class AccuracySummary:
    """Mean accuracy and a document-level bootstrap interval."""

    n_rows: int
    n_docs: int
    accuracy: float
    ci_low: float
    ci_high: float


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean, or zero for empty inputs."""

    return sum(values) / len(values) if values else 0.0


def _quantile(values: Sequence[float], q: float) -> float:
    """Return a deterministic linear-interpolated quantile."""

    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def accuracy_summary(rows: Iterable[AccuracyRow], *, n_bootstrap: int = N_BOOTSTRAP,
                     seed: int = BOOTSTRAP_SEED) -> AccuracySummary:
    """Return mean accuracy and a document-level bootstrap 95% CI.

    The point estimate is the row-level mean over all questions in `rows`.
    Bootstrap samples draw document ids with replacement and include all rows
    from each sampled document, preserving within-document correlation.
    """

    materialized = list(rows)
    if not materialized:
        return AccuracySummary(n_rows=0, n_docs=0, accuracy=0.0, ci_low=0.0, ci_high=0.0)

    by_doc: dict[str, list[float]] = defaultdict(list)
    for row in materialized:
        by_doc[row.doc_id].append(1.0 if row.correct else 0.0)

    doc_ids = sorted(by_doc)
    accuracy = _mean([value for values in by_doc.values() for value in values])
    if len(doc_ids) == 1 or n_bootstrap <= 0:
        return AccuracySummary(len(materialized), len(doc_ids), accuracy, accuracy, accuracy)

    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_bootstrap):
        sampled_values: list[float] = []
        for _ in doc_ids:
            sampled_values.extend(by_doc[rng.choice(doc_ids)])
        samples.append(_mean(sampled_values))

    return AccuracySummary(
        n_rows=len(materialized),
        n_docs=len(doc_ids),
        accuracy=accuracy,
        ci_low=_quantile(samples, BOOTSTRAP_CI_LOW),
        ci_high=_quantile(samples, BOOTSTRAP_CI_HIGH),
    )
