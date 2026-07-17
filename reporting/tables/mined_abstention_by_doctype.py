"""Mined: abstention rate on unanswerable questions by prompt mode AND doc_type.

The planned hallucination table quantifies the targeted-prompt effect overall; this
adds the doc_type axis to show where abstention behaviour varies across document
types.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._common import Table, doc_type_of, group_by, ordered_doc_types
from ._load import column_n_footer
from .hallucination import _MODE_ORDER, prompt_mode_of


def build(rows: Sequence[Any]) -> Table:
    """prompt_mode x doc_type -> abstention rate over unanswerable questions."""

    unanswerable = [r for r in rows if getattr(r, "is_unanswerable", False)] or list(rows)
    modes = sorted({prompt_mode_of(r) for r in unanswerable}, key=lambda m: (_MODE_ORDER.get(m, 99), m))

    columns = ["doc_type", *modes, *(f"n_{m}" for m in modes)]
    by_doc_type = group_by(unanswerable, doc_type_of)
    table_rows: list[list[str]] = []
    for dt in ordered_doc_types(unanswerable):
        by_mode = group_by(by_doc_type[dt], prompt_mode_of)
        rates: list[str] = []
        counts: list[str] = []
        for mode in modes:
            group = by_mode.get(mode, [])
            abstained = sum(1 for r in group if getattr(r, "abstained", False))
            rates.append(f"{abstained / len(group) * 100:.1f}" if group else "-")
            counts.append(str(len(group)))
        table_rows.append([dt, *rates, *counts])
    n_by_col = {mode: sum(1 for r in unanswerable if prompt_mode_of(r) == mode) for mode in modes}
    return Table(
        key="mined_abstention_by_doctype",
        title="Mined: abstention rate on unanswerable questions by prompt mode and doc_type",
        columns=columns,
        rows=table_rows,
        footer=column_n_footer(columns, n_by_col),
    )
