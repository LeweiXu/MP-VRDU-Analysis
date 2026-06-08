"""Answer extraction + normalisation (the heart of MMLongBench-Doc scoring).

MMLongBench-Doc scoring is NOT exact match (context.md §11): the official
pipeline uses an LLM to extract a short answer from the model response, then a
rule-based comparison that is `answer_format`-aware (Int / Float / Str / List /
None). This module implements the normalisation + comparison rules; the LLM
extraction step is optional and lives behind the Judge interface.
"""

from __future__ import annotations

import re
import string
from typing import Optional

# Phrases that signal the model declined to answer (the unanswerable class).
_ABSTAIN_PATTERNS = [
    "not answerable", "cannot answer", "can't answer", "unanswerable",
    "no answer", "not enough information", "insufficient information",
    "cannot be determined", "not provided", "not mentioned", "n/a",
    "not available", "no information",
]

_ARTICLES = {"a", "an", "the"}


def is_abstention(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    return any(p in t for p in _ABSTAIN_PATTERNS)


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation + articles, collapse whitespace."""
    t = (text or "").lower().strip()
    t = t.replace("%", " percent")
    t = "".join(ch if ch not in string.punctuation else " " for ch in t)
    tokens = [w for w in t.split() if w not in _ARTICLES]
    return " ".join(tokens)


def extract_number(text: str) -> Optional[float]:
    """Pull the first numeric value out of a string (handles commas, %, $)."""
    if text is None:
        return None
    cleaned = str(text).replace(",", "").replace("$", "").replace("%", "")
    m = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    return float(m.group()) if m else None


def normalize_list(text: str) -> list[str]:
    """Split a list-style answer into normalised elements."""
    if text is None:
        return []
    raw = re.split(r"[;,/\n]| and ", str(text))
    return sorted({normalize_text(x) for x in raw if normalize_text(x)})


def list_f1(pred: list[str], gold: list[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    pset, gset = set(pred), set(gold)
    tp = len(pset & gset)
    if tp == 0:
        return 0.0
    precision = tp / len(pset)
    recall = tp / len(gset)
    return 2 * precision * recall / (precision + recall)


def compare(pred: str, gold: str, answer_format: str,
            float_rtol: float = 0.01, list_f1_threshold: float = 0.999) -> bool:
    """Rule-based correctness for one (pred, gold) pair given the answer format.

    Returns True/False. Abstention handling is done by the caller
    (score_answer) because it needs to know the gold's answerability.
    """
    fmt = (answer_format or "Str").strip().lower()

    if fmt in {"int", "integer"}:
        pn, gn = extract_number(pred), extract_number(gold)
        return pn is not None and gn is not None and int(round(pn)) == int(round(gn))

    if fmt in {"float", "double", "number"}:
        pn, gn = extract_number(pred), extract_number(gold)
        if pn is None or gn is None:
            return False
        if gn == 0:
            return abs(pn) < 1e-6
        return abs(pn - gn) / abs(gn) <= float_rtol

    if fmt in {"list", "array"}:
        return list_f1(normalize_list(pred), normalize_list(gold)) >= list_f1_threshold

    # Str (default): normalised equality, with substring + numeric fallbacks.
    np_, ng = normalize_text(pred), normalize_text(gold)
    if np_ == ng:
        return True
    if ng and (ng in np_ or np_ in ng) and len(ng) > 2:
        return True
    pn, gn = extract_number(pred), extract_number(gold)
    if pn is not None and gn is not None:
        return abs(pn - gn) < 1e-6
    return False
