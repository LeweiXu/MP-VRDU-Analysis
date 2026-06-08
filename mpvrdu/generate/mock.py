"""MockGenerator — canned outputs for local plumbing tests (no model).

Modes:
- "gold":  return the gold answer for the question (needs a qid->answer lookup).
           Proves a *correct* generator scores ~100% -> validates the SCORING.
- "wrong": return a fixed incorrect string -> validates a wrong generator ~0%.
- "echo":  return a snippet of the provided text/image count -> exercises the
           input-builder plumbing without needing gold.

NEVER report numbers from the mock — it exists only to test the harness.
"""

from __future__ import annotations

from typing import Optional

from .base import Generator

WRONG_ANSWER = "completely-incorrect-sentinel-zzzz"


class MockGenerator(Generator):
    name = "mock"

    def __init__(self, mode: str = "gold", gold_lookup: Optional[dict] = None,
                 current_qid: Optional[str] = None):
        self.mode = mode
        self.gold_lookup = gold_lookup or {}
        # the pipeline sets current_qid before each call so "gold" knows which Q
        self.current_qid = current_qid

    def set_qid(self, qid: str) -> None:
        self.current_qid = qid

    def answer(self, question: str, images: Optional[list[str]] = None,
               text: Optional[str] = None) -> str:
        if self.mode == "gold":
            if self.current_qid is None or self.current_qid not in self.gold_lookup:
                raise RuntimeError(
                    "MockGenerator(gold) needs current_qid set to a known qid; "
                    f"got {self.current_qid!r}"
                )
            return self.gold_lookup[self.current_qid]
        if self.mode == "wrong":
            return WRONG_ANSWER
        # echo
        n_img = len(images or [])
        snippet = (text or "")[:60].replace("\n", " ")
        return f"[echo] images={n_img} text={snippet!r}"
