"""Top-k retrieval sweep: accuracy as k grows, per retrieval modality."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._common import Table, acc_cell, group_by
from .matched_cross import parse_condition


def build(rows: Sequence[Any]) -> Table:
    """One row per k: accuracy for each retrieval modality at that depth."""

    tagged = []
    for row in rows:
        parsed = parse_condition(getattr(row, "condition", ""))
        if parsed:
            tagged.append((row, parsed[0], parsed[1]))

    modalities = [m for m in ("text", "vision", "joint") if any(mm == m for _, mm, _ in tagged)]
    ks = sorted({k for _, _, k in tagged})
    columns = ["k", *modalities, "n"]
    by_k = group_by(tagged, lambda t: t[2])
    table_rows: list[list[str]] = []
    for k in ks:
        at_k = by_k[k]
        cells = [acc_cell([r for r, m, _ in at_k if m == modality]) for modality in modalities]
        table_rows.append([str(k), *cells, str(len(at_k))])
    return Table(
        key="kdepth",
        title="Top-k sweep: accuracy vs retrieval depth by modality",
        columns=columns,
        rows=table_rows,
    )
