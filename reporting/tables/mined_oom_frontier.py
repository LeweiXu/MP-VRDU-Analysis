"""Mined: OOM rate by rung, resolution, and pages-fed (the empirical "what fits" map).

Built from the status rows in predictions.jsonl (no judge needed): where cells OOM'd
on the 16 GB V100 is a deployment finding hiding in the failure rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from scoring.frontier import RUNG_ORDER

from ._common import Table, group_by
from ._load import column_n_footer

# Page-count buckets keep per-cell n large enough to read a rate.
_PAGE_BUCKETS = ((1, 1, "1"), (2, 5, "2-5"), (6, 10, "6-10"), (11, 20, "11-20"), (21, 10**9, "21+"))


def pages_bucket(row: Any) -> str:
    pages = getattr(row, "page_indices", None) or []
    n = len(pages)
    for lo, hi, label in _PAGE_BUCKETS:
        if lo <= n <= hi:
            return label
    return "0"


def _rung_rank(rung: str) -> int:
    return RUNG_ORDER.index(rung) if rung in RUNG_ORDER else len(RUNG_ORDER)


def build(rows: Sequence[Any]) -> Table:
    """(rung, resolution, pages-fed) -> OOM rate over all cells in the group."""

    columns = ["rung", "resolution", "pages_fed", "oom_rate", "n_oom", "n_total"]

    def key(r: Any) -> tuple[str, str, str]:
        return (getattr(r, "representation", ""), getattr(r, "visual_resolution", ""), pages_bucket(r))

    def sort_key(k: tuple[str, str, str]) -> tuple:
        rung, res, bucket = k
        order = next((i for i, (_, _, lab) in enumerate(_PAGE_BUCKETS) if lab == bucket), 99)
        return (_rung_rank(rung), res, order)

    table_rows: list[list[str]] = []
    grouped = group_by(rows, key)
    for k in sorted(grouped, key=sort_key):
        group = grouped[k]
        n_oom = sum(1 for r in group if getattr(r, "status", "") == "oom")
        rate = n_oom / len(group) if group else 0.0
        rung, res, bucket = k
        table_rows.append([rung, res, bucket, f"{rate * 100:.1f}", str(n_oom), str(len(group))])
    return Table(
        key="mined_oom_frontier",
        title="Mined: OOM rate by rung, resolution, and pages-fed",
        columns=columns,
        rows=table_rows,
        note="rate = oom cells / all cells in the group, over the 16 GB V100 runs.",
        footer=column_n_footer(columns, {}),
    )


def summary(rows: Sequence[Any]) -> Table:
    """Overall OOM rate by rung × resolution, pooled over page-count buckets."""

    usable = list(rows)
    seen_res = sorted({getattr(r, "visual_resolution", "") for r in usable})
    present_rungs = [r for r in RUNG_ORDER if any(getattr(x, "representation", "") == r for x in usable)]
    columns = ["rung", *seen_res, "n_total"]
    n_by_col = {res: 0 for res in seen_res}
    table_rows: list[list[str]] = []
    for rung in present_rungs:
        rung_rows = [r for r in usable if getattr(r, "representation", "") == rung]
        by_res = group_by(rung_rows, lambda r: getattr(r, "visual_resolution", ""))
        cells: list[str] = []
        for res in seen_res:
            group = by_res.get(res, [])
            n_by_col[res] += len(group)
            if group:
                rate = sum(1 for r in group if getattr(r, "status", "") == "oom") / len(group)
                cells.append(f"{rate * 100:.1f}")
            else:
                cells.append("-")
        table_rows.append([rung, *cells, str(len(rung_rows))])
    return Table(key="oom_frontier_summary", title="OOM frontier (overall): OOM rate by rung and resolution",
                 columns=columns, rows=table_rows, footer=column_n_footer(columns, n_by_col),
                 note="rate = oom cells / all cells; pooled over page buckets and G1 runs.")
