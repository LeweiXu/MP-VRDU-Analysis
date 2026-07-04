"""Detect abstention/refusal answers for unanswerable-question analysis.

Purpose:
    Centralises the pre-registered surface-form definition of "the model
    declined to answer" so judges and later hallucination metrics do not drift.

Pipeline role:
    `StubJudge` already uses `is_abstention()`; later metrics will reuse it for
    native-unanswerable questions and retrieval-miss hallucination rates.

Arguments:
    None. This module is import-only; callers pass prediction text to
    `is_abstention(text)`.
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
