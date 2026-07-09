"""Retrieval matched-versus-cross: accuracy by retrieval modality and bin, at the
TLV reasoning rung."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from ._common import Table, acc_cell, bin_of, group_by, ordered_bins

_COND = re.compile(r"(?:retrieved|similarity)_(?P<modality>text|vision|joint|bm25|bge|colqwen\w*)_k(?P<k>\d+)")

# Condition tokens that name a text vs a vision retriever, mapped to the modality.
_MODALITY = {"text": "text", "bm25": "text", "bge": "text", "vision": "vision", "joint": "joint"}


def parse_condition(cond: str) -> tuple[str, int] | None:
    """Return `(modality, k)` from a retrieval condition, or None."""

    m = _COND.match(cond or "")
    if not m:
        return None
    token = m.group("modality")
    modality = _MODALITY.get(token, "vision" if token.startswith("colqwen") else token)
    return modality, int(m.group("k"))


def modality_of(row: Any) -> str | None:
    parsed = parse_condition(getattr(row, "condition", ""))
    return parsed[0] if parsed else None


def build(rows: Sequence[Any]) -> Table:
    """Bin x retrieval-modality accuracy (retrieval modality vs reasoning at TLV)."""

    tagged = [(r, modality_of(r)) for r in rows]
    tagged = [(r, m) for r, m in tagged if m is not None]
    modalities = [m for m in ("text", "vision", "joint") if any(mm == m for _, mm in tagged)]
    columns = ["bin", *modalities, "n"]
    by_bin = group_by([r for r, _ in tagged], bin_of)
    table_rows: list[list[str]] = []
    for b in ordered_bins([r for r, _ in tagged]):
        bin_rows = by_bin[b]
        cells = [acc_cell([r for r in bin_rows if modality_of(r) == m]) for m in modalities]
        table_rows.append([b, *cells, str(len(bin_rows))])
    return Table(
        key="matched_cross",
        title="Matched vs cross: accuracy by retrieval modality and bin (TLV)",
        columns=columns,
        rows=table_rows,
    )
