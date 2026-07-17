"""Cost-ordered representation ladder by doc_type: the headline accuracy frontier."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from scoring.frontier import RUNG_ORDER

from ._common import (
    Table,
    acc_cell,
    base_condition,
    doc_type_of,
    frontier_rung,
    group_by,
    ordered_doc_types,
    restrict_to_primary_spec,
)
from ._load import column_n_footer


def _present_rungs(rows: Sequence[Any], allowed: Sequence[str]) -> list[str]:
    seen = {getattr(r, "representation", "") for r in rows}
    return [r for r in RUNG_ORDER if r in allowed and r in seen]


def ladder_by_doc_type(
    rows: Sequence[Any],
    *,
    key: str,
    title: str,
    rungs: Sequence[str] = RUNG_ORDER,
    margin_points: float = 3.0,
    with_frontier: bool = True,
    note: str = "",
) -> Table:
    """doc_type x rung accuracy grid, one row per doc_type, optional frontier column."""

    present = _present_rungs(rows, rungs)
    columns = ["doc_type", *present]
    if with_frontier:
        columns.append("frontier")
    columns.append("n")
    by_doc_type = group_by(rows, doc_type_of)
    by_rung_all = group_by(rows, lambda r: getattr(r, "representation", ""))
    table_rows: list[list[str]] = []
    for dt in ordered_doc_types(rows):
        dt_rows = by_doc_type[dt]
        by_rung = group_by(dt_rows, lambda r: getattr(r, "representation", ""))
        cells = [acc_cell(by_rung.get(rung, [])) for rung in present]
        row = [dt, *cells]
        if with_frontier:
            row.append(frontier_rung(dt_rows, margin_points=margin_points) or "-")
        row.append(str(len(dt_rows)))
        table_rows.append(row)
    footer = column_n_footer(columns, {rung: len(by_rung_all.get(rung, [])) for rung in present})
    return Table(key=key, title=title, columns=columns, rows=table_rows, note=note, footer=footer)


def build(rows: Sequence[Any], *, margin_points: float = 3.0) -> Table:
    """Headline table: oracle-page accuracy across the four rungs, per doc_type."""

    oracle = [r for r in rows if base_condition(getattr(r, "condition", "")) == "oracle"]
    return ladder_by_doc_type(
        restrict_to_primary_spec(oracle or rows),
        key="headline",
        title="Headline: cost-ordered ladder accuracy by doc_type (oracle pages)",
        margin_points=margin_points,
    )


def summary(rows: Sequence[Any], *, margin_points: float = 3.0) -> Table:
    """Overall ladder accuracy pooled across all doc_types (one row)."""

    oracle = [r for r in rows if base_condition(getattr(r, "condition", "")) == "oracle"]
    pooled = restrict_to_primary_spec(oracle or list(rows))
    present = _present_rungs(pooled, RUNG_ORDER)
    by_rung = group_by(pooled, lambda r: getattr(r, "representation", ""))
    columns = ["scope", *present, "frontier", "n"]
    row = ["all doc_types", *[acc_cell(by_rung.get(rung, [])) for rung in present],
           frontier_rung(pooled, margin_points=margin_points) or "-", str(len(pooled))]
    footer = column_n_footer(columns, {rung: len(by_rung.get(rung, [])) for rung in present})
    return Table(key="headline_summary", title="Headline (overall): ladder accuracy across all doc_types",
                 columns=columns, rows=[row], footer=footer)
