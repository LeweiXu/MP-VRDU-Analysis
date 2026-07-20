"""Multi-page integration: oracle accuracy split by evidence hop (single- vs
multi-page), per rung and doc_type, with the single-minus-multi gap."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from scoring.accuracy import accuracy_summary
from scoring.frontier import RUNG_ORDER

from ._common import (
    Table,
    acc_cell,
    doc_type_of,
    group_by,
    ordered_doc_types,
    restrict_to_primary_spec,
    rows_for_condition,
)
from ._load import column_n_footer

# The answerable pool splits into single- and multi-page evidence. `hop=none` is a
# third value on the row but is dropped here: on this pool it is not the
# unanswerable questions (every such row carries is_unanswerable=false) but the
# handful of answerable questions that recorded no gold evidence pages, so it has
# no integration reading.
HOPS = ("single", "multi")
GAP_COLUMN = "M − S"
NOTE = (
    "hop=none is dropped: those rows are answerable questions that recorded no gold "
    "evidence pages, not unanswerable ones, so they carry no integration signal. "
    "Accuracy columns are percentages. `M − S` is multi-page accuracy MINUS "
    "single-page accuracy, in points, so it reads as how multi-page evidence performs "
    "relative to single-page: a NEGATIVE value means multi-page is worse."
)


def _hop_of(row: Any) -> str:
    return getattr(row, "hop", "") or ""


def _answerable_hops(rows: Sequence[Any]) -> list[Any]:
    """Oracle rows restricted to the primary reasoner and to a real hop level."""

    oracle = restrict_to_primary_spec(rows_for_condition(rows, "oracle"))
    return [r for r in oracle if _hop_of(r) in HOPS]


def _present_rungs(rows: Sequence[Any]) -> list[str]:
    seen = {getattr(r, "representation", "") for r in rows}
    return [r for r in RUNG_ORDER if r in seen]


def _gap_cell(by_hop: dict[str, list[Any]]) -> str:
    """Multi minus single accuracy in points, or `-` when either side is empty."""

    single, multi = by_hop.get("single", []), by_hop.get("multi", [])
    if not single or not multi:
        return "-"
    delta = accuracy_summary(multi).accuracy - accuracy_summary(single).accuracy
    return f"{delta * 100:+.1f}"


def _hop_row(label: list[str], rows: Sequence[Any]) -> list[str]:
    """One grid row: the leading labels, accuracy per hop, the gap, and n."""

    by_hop = group_by(rows, _hop_of)
    cells = [acc_cell(by_hop.get(hop, [])) for hop in HOPS]
    return [*label, *cells, _gap_cell(by_hop), str(len(rows))]


def _footer(columns: Sequence[str], rows: Sequence[Any]) -> list[list[str]]:
    by_hop = group_by(rows, _hop_of)
    return column_n_footer(columns, {hop: len(by_hop.get(hop, [])) for hop in HOPS})


def build(rows: Sequence[Any]) -> Table:
    """Accuracy by hop for each doc_type and rung (oracle pages)."""

    scoped = _answerable_hops(rows)
    present = _present_rungs(scoped)
    columns = ["doc_type", "rung", *HOPS, GAP_COLUMN, "n"]
    by_doc_type = group_by(scoped, doc_type_of)

    table_rows: list[list[str]] = []
    for dt in ordered_doc_types(scoped):
        by_rung = group_by(by_doc_type[dt], lambda r: getattr(r, "representation", ""))
        for rung in present:
            rung_rows = by_rung.get(rung, [])
            if rung_rows:
                table_rows.append(_hop_row([dt, rung], rung_rows))

    return Table(
        key="integration",
        title="Integration: accuracy by evidence hop, per doc_type and rung (oracle pages)",
        columns=columns,
        rows=table_rows,
        note=NOTE,
        footer=_footer(columns, scoped),
    )


def summary(rows: Sequence[Any]) -> Table:
    """Accuracy by hop per rung, pooled across all doc_types (one row per rung)."""

    scoped = _answerable_hops(rows)
    columns = ["rung", *HOPS, GAP_COLUMN, "n"]
    by_rung = group_by(scoped, lambda r: getattr(r, "representation", ""))
    table_rows = [_hop_row([rung], by_rung[rung]) for rung in _present_rungs(scoped)]
    return Table(
        key="integration_summary",
        title="Integration (overall): accuracy by evidence hop per rung",
        columns=columns,
        rows=table_rows,
        note=NOTE,
        footer=_footer(columns, scoped),
    )
