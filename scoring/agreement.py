"""Judge-human agreement (Cohen's kappa)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

AGREEMENT_LABELS: tuple[str, ...] = ("correct", "incorrect", "abstained")
KAPPA_THRESHOLD = 0.75


def result_row_label(row) -> str:
    """Return the agreement label implied by a judged result row."""

    if getattr(row, "abstained", False):
        return "abstained"
    return "correct" if getattr(row, "correct", False) else "incorrect"


def normalise_agreement_label(value: str) -> str:
    """Normalise a human or judge label to the agreement label set."""

    label = " ".join(str(value).strip().replace("_", " ").split()).casefold()
    aliases = {
        "correct": "correct",
        "right": "correct",
        "yes": "correct",
        "incorrect": "incorrect",
        "wrong": "incorrect",
        "no": "incorrect",
        "abstained": "abstained",
        "abstain": "abstained",
        "refusal": "abstained",
    }
    try:
        return aliases[label]
    except KeyError as exc:
        raise ValueError(f"unknown agreement label {value!r}") from exc


def cohen_kappa(
    judge_labels: Sequence[str],
    human_labels: Sequence[str],
    *,
    labels: Sequence[str] = AGREEMENT_LABELS,
) -> float:
    """Compute Cohen's kappa for two equal-length label sequences."""

    if len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels and human_labels must have the same length")
    if not judge_labels:
        raise ValueError("at least one labelled item is required")

    normal_judge = [normalise_agreement_label(label) for label in judge_labels]
    normal_human = [normalise_agreement_label(label) for label in human_labels]
    n = len(normal_judge)
    observed = sum(a == b for a, b in zip(normal_judge, normal_human, strict=True)) / n
    judge_counts = Counter(normal_judge)
    human_counts = Counter(normal_human)
    expected = sum((judge_counts[label] / n) * (human_counts[label] / n) for label in labels)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def agreement_passes(kappa: float, *, threshold: float = KAPPA_THRESHOLD) -> bool:
    """Return whether judge-human kappa meets the pre-registered threshold."""

    return float(kappa) >= float(threshold)
