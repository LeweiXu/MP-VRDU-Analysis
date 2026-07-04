"""Abstention and hallucination metrics for unanswerable or missed-evidence cases.

Stage 7 fills in the aggregate abstention-rate / hallucination-rate metrics
(RQ5). Stage 3 defines only `is_abstention`, the pre-registered refusal-surface
test from the Stage 1 checkpoint, because the stub judge and later metrics all
need the same definition of "the model declined to answer".
"""

from __future__ import annotations

import re

#: Normalised refusal / no-evidence surface forms (Stage 1 abstention definition).
ABSTENTION_FORMS: tuple[str, ...] = (
    "not answerable",
    "cannot be answered",
    "can not be answered",
    "cannot answer",
    "unanswerable",
    "insufficient information",
    "not enough information",
    "no answer",
    "unknown from the document",
    "not mentioned",
    "not provided",
)


def is_abstention(text: str) -> bool:
    """Return whether a prediction contains a refusal / no-evidence surface form."""

    normalised = re.sub(r"\s+", " ", str(text).strip().casefold())
    return any(form in normalised for form in ABSTENTION_FORMS)
