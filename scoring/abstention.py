"""Detects abstention in reasoner answers."""

from __future__ import annotations

import re

from config import ABSTENTION_FORMS


def is_abstention(text: str) -> bool:
    """Return whether a prediction contains a refusal / no-evidence surface form."""

    normalised = re.sub(r"\s+", " ", str(text).strip().casefold())
    return any(form in normalised for form in ABSTENTION_FORMS)
