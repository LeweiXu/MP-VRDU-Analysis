"""Evidence-source composition (appendix): accuracy per rung split by the
evidence modality a question draws on."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from scoring.frontier import RUNG_ORDER

from ._common import Table, acc_cell, group_by, restrict_to_primary_spec


def build(rows: Sequence[Any]) -> Table:
    """One row per evidence source: accuracy across rungs.

    A question can cite several sources (e.g. Chart + Table), so it contributes to
    each source's row; the row count reflects those repeats.
    """

    oracle = restrict_to_primary_spec(
        [r for r in rows if getattr(r, "condition", "") == "oracle"] or list(rows)
    )
    present = [r for r in RUNG_ORDER if any(getattr(x, "representation", "") == r for x in oracle)]

    by_source: dict[str, list[Any]] = {}
    for row in oracle:
        sources = getattr(row, "evidence_sources", ()) or ("(none)",)
        for source in sources:
            by_source.setdefault(str(source), []).append(row)

    columns = ["evidence_source", *present, "n"]
    table_rows: list[list[str]] = []
    for source in sorted(by_source):
        source_rows = by_source[source]
        by_rung = group_by(source_rows, lambda r: getattr(r, "representation", ""))
        cells = [acc_cell(by_rung.get(rung, [])) for rung in present]
        table_rows.append([source, *cells, str(len(source_rows))])
    return Table(
        key="composition",
        title="Composition: accuracy by evidence source and rung (appendix)",
        columns=columns,
        rows=table_rows,
    )
