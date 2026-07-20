"""Oracle accuracy per rung against the number of gold evidence pages a question
cites, bucketed, as the finer-grained companion to the single/multi hop split."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from scoring.frontier import RUNG_ORDER

from ._common import Table, acc_cell, group_by, restrict_to_primary_spec, rows_for_condition
from ._load import column_n_footer

# Ordered buckets over the gold evidence-page count. The raw counts have a long thin
# tail (40 questions at 3 pages, then 36 across 4-5 and 31 across 6-24), so the tail
# is bucketed rather than shown per page count.
BUCKETS: tuple[str, ...] = ("1", "2", "3", "4-5", "6+")
SLOPE_COLUMN = "1 → 6+"
NOTE = (
    "Rows are the number of gold evidence pages the question cites, taken from the "
    "corpus annotation that `hop` itself is derived from (NOT from `page_indices`, "
    "which for a no-gold-page question carries a stand-in page and would misbucket "
    "it). Questions with zero evidence pages are excluded. "
    "The 4-5 and 6+ buckets are small (about 36 and 31 questions before OOM "
    "attrition, against 480 at one page), so read them for trend, not precision: "
    "check the per-cell n before quoting either. "
    "This is the finer-grained companion to the integration table, which keeps the "
    "collapsed single-versus-multi view."
)


@lru_cache(maxsize=1)
def _evidence_page_counts() -> dict[str, int]:
    """question_id -> gold evidence-page count, from the corpus.

    The result row carries `hop` (none/single/multi) but not the underlying count,
    and `page_indices` is the pages actually fed rather than the gold set, so the
    corpus is the only place the exact count lives.
    """

    from config import ExperimentConfig
    from data.loader import load_mmlongbench

    data_dir = getattr(getattr(ExperimentConfig(), "paths", None), "data_dir", None)
    return {q.id: len(q.evidence_pages) for q in load_mmlongbench(data_dir)}


def _bucket_of(row: Any) -> str:
    """The row's evidence-page bucket, or `""` when it has no gold pages."""

    count = _evidence_page_counts().get(getattr(row, "question_id", ""), 0)
    if count <= 0:
        return ""
    if count <= 3:
        return str(count)
    return "4-5" if count <= 5 else "6+"


def _scoped(rows: Sequence[Any]) -> list[Any]:
    """Oracle rows for the primary reasoner that carry at least one gold page."""

    oracle = restrict_to_primary_spec(rows_for_condition(rows, "oracle"))
    return [r for r in oracle if _bucket_of(r)]


def _present(rows: Sequence[Any], values: Sequence[str], keyfn) -> list[str]:
    seen = {keyfn(r) for r in rows}
    return [v for v in values if v in seen]


def build(rows: Sequence[Any]) -> Table:
    """Accuracy per rung for each evidence-page bucket (oracle pages)."""

    scoped = _scoped(rows)
    rungs = _present(scoped, RUNG_ORDER, lambda r: getattr(r, "representation", ""))
    buckets = _present(scoped, BUCKETS, _bucket_of)
    by_bucket = group_by(scoped, _bucket_of)

    columns = ["evidence pages", *rungs, "n"]
    table_rows: list[list[str]] = []
    for bucket in buckets:
        bucket_rows = by_bucket[bucket]
        by_rung = group_by(bucket_rows, lambda r: getattr(r, "representation", ""))
        # n per cell, not just per row: OOM attrition differs sharply by rung.
        cells = [f"{acc_cell(by_rung.get(rung, []))} (n={len(by_rung.get(rung, []))})" for rung in rungs]
        table_rows.append([bucket, *cells, str(len(bucket_rows))])

    n_by_col = {rung: sum(1 for r in scoped if getattr(r, "representation", "") == rung) for rung in rungs}
    return Table(
        key="hop_rung",
        title="Integration detail: accuracy by gold evidence-page count and rung (oracle pages)",
        columns=columns,
        rows=table_rows,
        note=NOTE,
        footer=column_n_footer(columns, n_by_col),
    )


def summary(rows: Sequence[Any]) -> Table:
    """Per rung, accuracy across the buckets plus the drop from the 1-page bucket."""

    scoped = _scoped(rows)
    rungs = _present(scoped, RUNG_ORDER, lambda r: getattr(r, "representation", ""))
    buckets = _present(scoped, BUCKETS, _bucket_of)
    by_rung = group_by(scoped, lambda r: getattr(r, "representation", ""))

    columns = ["rung", *buckets, SLOPE_COLUMN, "n"]
    table_rows: list[list[str]] = []
    for rung in rungs:
        rung_rows = by_rung[rung]
        by_bucket = group_by(rung_rows, _bucket_of)
        # Per-cell n, not just the pooled column n: OOM attrition is far worse on the
        # image rungs at high page counts, so a rung can look robust here purely
        # because only its easiest few questions survived to be scored.
        cells = [f"{acc_cell(by_bucket.get(bucket, []))} (n={len(by_bucket.get(bucket, []))})"
                 for bucket in buckets]
        table_rows.append([rung, *cells, _slope(by_bucket, buckets), str(len(rung_rows))])

    n_by_col = {bucket: sum(1 for r in scoped if _bucket_of(r) == bucket) for bucket in buckets}
    return Table(
        key="hop_rung_summary",
        title="Integration detail (overall): how each rung degrades with evidence-page count",
        columns=columns,
        rows=table_rows,
        note=NOTE + f" `{SLOPE_COLUMN}` is the accuracy of the last bucket minus the first, "
                    "in points; negative means the rung degrades as evidence spreads. "
                    "Read it against the per-cell n: TLV OOMs hardest at high page counts, "
                    "so its tail buckets are a handful of surviving questions and its slope "
                    "is not comparable to the text rungs'.",
        footer=column_n_footer(columns, n_by_col),
    )


def _slope(by_bucket: dict[str, list[Any]], buckets: Sequence[str]) -> str:
    """Accuracy of the widest bucket minus the narrowest, in points."""

    from scoring.accuracy import accuracy_summary

    first, last = by_bucket.get(buckets[0], []), by_bucket.get(buckets[-1], [])
    if not first or not last:
        return "-"
    delta = accuracy_summary(last).accuracy - accuracy_summary(first).accuracy
    return f"{delta * 100:+.1f}"
