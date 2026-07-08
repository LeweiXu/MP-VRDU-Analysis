"""Deduplicated union of a text page set and a vision page set."""

from __future__ import annotations

from collections.abc import Sequence


def union(*page_sets: Sequence[int], k: int | None = None) -> tuple[int, ...]:
    """Return the order-preserving deduplicated union of page sets.

    Joint retrieval is free: it combines page sets already produced by a text and
    a vision retriever, with no new retrieval and no score fusion. Pages keep
    first-seen order across the inputs; `k` optionally caps the result.
    """

    seen: list[int] = []
    for page_set in page_sets:
        for page in page_set:
            page = int(page)
            if page not in seen:
                seen.append(page)
    return tuple(seen[:k]) if k is not None else tuple(seen)
