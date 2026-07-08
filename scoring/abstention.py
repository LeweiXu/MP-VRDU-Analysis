"""Detects abstention in reasoner answers."""

from __future__ import annotations

import re

#: Normalised refusal / no-evidence surface forms.
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
