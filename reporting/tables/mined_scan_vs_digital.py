"""Mined: ladder accuracy for scanned vs digital documents, per doc_type and rung.

Tests the deployment hypothesis that the cheap `T` rung collapses on scans (embedded
text is empty by design) so the sufficiency frontier shifts toward `TLV`/`V`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from scoring.frontier import RUNG_ORDER

from ._common import Table, acc_cell, base_condition, doc_type_of, group_by, ordered_doc_types, restrict_to_primary_spec

# Digital first, then scanned; anything else sorts after.
_SCAN_ORDER = {"digital": 0, "scanned": 1}


def scan_label_of(row: Any) -> str:
    return getattr(row, "scan_label", "") or "(unlabeled)"


def build(rows: Sequence[Any]) -> Table:
    """doc_type x rung accuracy split by scan_label (digital vs scanned)."""

    oracle = restrict_to_primary_spec([r for r in rows if base_condition(getattr(r, "condition", "")) == "oracle"] or list(rows))
    labels = sorted({scan_label_of(r) for r in oracle}, key=lambda s: (_SCAN_ORDER.get(s, 99), s))
    present_rungs = [r for r in RUNG_ORDER if any(getattr(x, "representation", "") == r for x in oracle)]

    columns = ["doc_type", "rung", *labels, *(f"n_{lab}" for lab in labels)]
    by_doc_type = group_by(oracle, doc_type_of)
    table_rows: list[list[str]] = []
    for dt in ordered_doc_types(oracle):
        dt_rows = by_doc_type[dt]
        for rung in present_rungs:
            rung_rows = [r for r in dt_rows if getattr(r, "representation", "") == rung]
            by_scan = group_by(rung_rows, scan_label_of)
            accs = [acc_cell(by_scan.get(lab, [])) for lab in labels]
            counts = [str(len(by_scan.get(lab, []))) for lab in labels]
            table_rows.append([dt, rung, *accs, *counts])
    return Table(
        key="mined_scan_vs_digital",
        title="Mined: ladder accuracy, scanned vs digital, by doc_type and rung",
        columns=columns,
        rows=table_rows,
        note="oracle pages, primary reasoner. Empty T on scans is by design (no embedded text).",
    )


def summary(rows: Sequence[Any]) -> Table:
    """Overall ladder accuracy split by scan_label, pooled across doc_types (rung × scan)."""

    oracle = restrict_to_primary_spec(
        [r for r in rows if base_condition(getattr(r, "condition", "")) == "oracle"] or list(rows)
    )
    labels = sorted({scan_label_of(r) for r in oracle}, key=lambda s: (_SCAN_ORDER.get(s, 99), s))
    present_rungs = [r for r in RUNG_ORDER if any(getattr(x, "representation", "") == r for x in oracle)]
    columns = ["rung", *labels, *(f"n_{lab}" for lab in labels)]
    table_rows: list[list[str]] = []
    for rung in present_rungs:
        rung_rows = [r for r in oracle if getattr(r, "representation", "") == rung]
        by_scan = group_by(rung_rows, scan_label_of)
        accs = [acc_cell(by_scan.get(lab, [])) for lab in labels]
        counts = [str(len(by_scan.get(lab, []))) for lab in labels]
        table_rows.append([rung, *accs, *counts])
    return Table(key="scan_vs_digital_summary", title="Scanned vs digital (overall): ladder accuracy by rung",
                 columns=columns, rows=table_rows, note="empty T on scans is by design (no embedded text).")
