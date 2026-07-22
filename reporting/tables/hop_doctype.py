"""Integration cross-tab: oracle accuracy per doc_type and rung against the gold
evidence-page count, collapsed to 1 / 2 / 3+ buckets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from scoring.frontier import RUNG_ORDER

from ._common import Table, acc_cell, doc_type_of, group_by, ordered_doc_types
from ._load import column_n_footer
from .hop_rung import _bucket_of as _fine_bucket_of
from .hop_rung import _scoped

# The fine 3 / 4-5 / 6+ tail buckets from hop_rung, merged: crossed with seven
# doc_types they fall to n=1 per cell, so the tail is one 3+ bucket here.
BUCKETS: tuple[str, ...] = ("1", "2", "3+")
COLLAPSED = {"3": "3+", "4-5": "3+", "6+": "3+"}
ALL_ROW = "**All**"
NOTE = (
    "Buckets are the number of gold evidence pages the question cites, taken from "
    "the corpus annotation that `hop` is derived from (NOT from `page_indices`, "
    "which for a no-gold-page question carries a stand-in page and would misbucket "
    "it); zero-evidence questions are dropped. 3+ merges the finer 3 / 4-5 / 6+ "
    "buckets of the integration-detail table. "
    "Every cell carries its own n: OOM attrition is rung-dependent at high page "
    "counts (worst on TLV), so a thin cell reads as survivorship, not robustness — "
    "check the n before quoting the cell. The bolded All rows pool every doc_type."
)


def _bucket_of(row: Any) -> str:
    """The row's collapsed evidence-page bucket (`1`/`2`/`3+`), `""` for no gold pages."""

    fine = _fine_bucket_of(row)
    return COLLAPSED.get(fine, fine)


def _grid_rows(label: str, rows: Sequence[Any], rungs: Sequence[str]) -> list[list[str]]:
    """One row per present rung for a doc_type block: accuracy (n=..) per bucket."""

    out: list[list[str]] = []
    by_rung = group_by(rows, lambda r: getattr(r, "representation", ""))
    for rung in rungs:
        rung_rows = by_rung.get(rung, [])
        if not rung_rows:
            continue
        by_bucket = group_by(rung_rows, _bucket_of)
        cells = [f"{acc_cell(by_bucket.get(b, []))} (n={len(by_bucket.get(b, []))})"
                 for b in BUCKETS]
        out.append([label, rung, *cells, str(len(rung_rows))])
    return out


def build(rows: Sequence[Any]) -> Table:
    """Accuracy per doc_type, rung, and collapsed evidence-page bucket (oracle pages)."""

    scoped = _scoped(rows)
    rungs = [r for r in RUNG_ORDER if r in {getattr(x, "representation", "") for x in scoped}]
    columns = ["doc_type", "rung", *BUCKETS, "n"]

    table_rows: list[list[str]] = []
    by_doc_type = group_by(scoped, doc_type_of)
    for dt in ordered_doc_types(scoped):
        table_rows += _grid_rows(dt, by_doc_type[dt], rungs)
    table_rows += _grid_rows(ALL_ROW, scoped, rungs)

    n_by_col = {b: sum(1 for r in scoped if _bucket_of(r) == b) for b in BUCKETS}
    return Table(
        key="hop_doctype",
        title="Integration cross-tab: accuracy by doc_type, rung, and evidence-page bucket (oracle pages)",
        columns=columns,
        rows=table_rows,
        note=NOTE,
        footer=column_n_footer(columns, n_by_col),
    )
