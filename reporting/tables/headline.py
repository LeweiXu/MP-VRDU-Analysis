"""Cost-ordered representation ladder by bin: the headline accuracy frontier."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from scoring.frontier import RUNG_ORDER

from ._common import Table, acc_cell, bin_of, frontier_rung, group_by, ordered_bins


def _present_rungs(rows: Sequence[Any], allowed: Sequence[str]) -> list[str]:
    seen = {getattr(r, "representation", "") for r in rows}
    return [r for r in RUNG_ORDER if r in allowed and r in seen]


def ladder_by_bin(
    rows: Sequence[Any],
    *,
    key: str,
    title: str,
    rungs: Sequence[str] = RUNG_ORDER,
    margin_points: float = 3.0,
    with_frontier: bool = True,
    note: str = "",
) -> Table:
    """Bin x rung accuracy grid, one row per bin, optional frontier column."""

    present = _present_rungs(rows, rungs)
    columns = ["bin", *present]
    if with_frontier:
        columns.append("frontier")
    columns.append("n")
    by_bin = group_by(rows, bin_of)
    table_rows: list[list[str]] = []
    for b in ordered_bins(rows):
        bin_rows = by_bin[b]
        by_rung = group_by(bin_rows, lambda r: getattr(r, "representation", ""))
        cells = [acc_cell(by_rung.get(rung, [])) for rung in present]
        row = [b, *cells]
        if with_frontier:
            row.append(frontier_rung(bin_rows, margin_points=margin_points) or "-")
        row.append(str(len(bin_rows)))
        table_rows.append(row)
    return Table(key=key, title=title, columns=columns, rows=table_rows, note=note)


def build(rows: Sequence[Any], *, margin_points: float = 3.0) -> Table:
    """Headline table: oracle-page accuracy across the four rungs, per bin."""

    oracle = [r for r in rows if getattr(r, "condition", "") == "oracle"]
    return ladder_by_bin(
        oracle or rows,
        key="headline",
        title="Headline: cost-ordered ladder accuracy by bin (oracle pages)",
        margin_points=margin_points,
    )
