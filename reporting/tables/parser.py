"""Parser comparison: TL/TLV accuracy by doc_type across the parsers under test.

Which parser produced the TL/TLV text is a per-run property (one parser per
run_tag), so the comparison merges several run_tags, each labelled by its parser.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._common import Table, acc_cell, base_condition, doc_type_of, group_by, ordered_doc_types
from ._load import column_n_footer

_RUNGS = ("TL", "TLV")


def _rung(row: Any) -> str:
    return getattr(row, "representation", "")


def build(labeled: Sequence[tuple[str, Sequence[Any]]], *, margin_points: float = 3.0) -> Table:
    """One column block per parser (TL, TLV), one row per doc_type.

    `labeled` pairs a parser label with that parser's rows (each from its own
    run_tag). Reads oracle TL/TLV cells; a per-column n footer shows how many cells
    backed each parser/rung (they differ where a parser OOM'd or failed to parse).
    """

    parsers: list[str] = []
    rows_by_parser: dict[str, list[Any]] = {}
    all_rows: list[Any] = []
    for label, rows in labeled:
        oracle = [r for r in rows if base_condition(getattr(r, "condition", "")) == "oracle" and _rung(r) in _RUNGS]
        if not oracle:
            continue
        parsers.append(label)
        rows_by_parser[label] = oracle
        all_rows += oracle

    columns = ["doc_type", *[f"{p} {rung}" for p in parsers for rung in _RUNGS], "n"]
    n_by_col = {
        f"{p} {rung}": len([r for r in rows_by_parser[p] if _rung(r) == rung])
        for p in parsers
        for rung in _RUNGS
    }
    table_rows: list[list[str]] = []
    for dt in ordered_doc_types(all_rows):
        cells: list[str] = []
        for p in parsers:
            by_rung = group_by([r for r in rows_by_parser[p] if doc_type_of(r) == dt], _rung)
            cells += [acc_cell(by_rung.get(rung, [])) for rung in _RUNGS]
        n_dt = len({getattr(r, "question_id", "") for r in all_rows if doc_type_of(r) == dt})
        table_rows.append([dt, *cells, str(n_dt)])

    return Table(
        key="parser",
        title="Parser comparison: TL/TLV accuracy by doc_type",
        columns=columns,
        rows=table_rows,
        footer=column_n_footer(columns, n_by_col),
    )


def summary(labeled: Sequence[tuple[str, Sequence[Any]]], *, margin_points: float = 3.0) -> Table:
    """Overall TL/TLV accuracy per parser, pooled across all doc_types (one row each)."""

    columns = ["parser", *_RUNGS, "n"]
    table_rows: list[list[str]] = []
    n_by_col = {rung: 0 for rung in _RUNGS}
    for label, rows in labeled:
        oracle = [r for r in rows if base_condition(getattr(r, "condition", "")) == "oracle" and _rung(r) in _RUNGS]
        if not oracle:
            continue
        by_rung = group_by(oracle, _rung)
        cells = [acc_cell(by_rung.get(rung, [])) for rung in _RUNGS]
        for rung in _RUNGS:
            n_by_col[rung] += len(by_rung.get(rung, []))
        table_rows.append([label, *cells, str(len({getattr(r, "question_id", "") for r in oracle}))])
    return Table(key="parser_summary", title="Parser comparison (overall): TL/TLV accuracy per parser",
                 columns=columns, rows=table_rows, footer=column_n_footer(columns, n_by_col))
