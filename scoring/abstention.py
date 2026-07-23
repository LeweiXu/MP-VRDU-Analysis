"""Detects abstention in reasoner answers and extracts delimited final answers."""

from __future__ import annotations

import re

from config import ABSTENTION_FORMS


def is_abstention(text: str) -> bool:
    """Return whether a prediction contains a refusal / no-evidence surface form."""

    normalised = re.sub(r"\s+", " ", str(text).strip().casefold())
    return any(form in normalised for form in ABSTENTION_FORMS)


def extract_final_answer(text: str, delimiter: str | None) -> str:
    """The text after the LAST occurrence of `delimiter`; the whole text otherwise.

    The prompt body's own generation cue is the same string as the usual
    delimiter ("Answer:"), and a reasoning-bearing answer may echo it, so only
    the last occurrence marks the final answer. No delimiter, or a delimiter the
    answer never emitted, returns the text unchanged (the whole generation goes
    to the judge, today's behaviour).
    """

    text = str(text)
    if not delimiter:
        return text
    _, found, tail = text.rpartition(delimiter)
    return tail.strip() if found else text
