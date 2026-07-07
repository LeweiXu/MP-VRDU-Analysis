"""Table 1: the headline frontier (doc-type bin x representation ladder).

Purpose:
    The core RQ1 table: for each Option-A bin, per-rung accuracy/cost and the
    sufficiency frontier (cheapest rung within the margin of the strongest). Fed
    by G1's oracle rows; Tables 6 and 7 reuse this builder to find the
    vision-benefit bins and the per-bin recipe.

Arguments:
    None. Import-only; `build_table1_headline(rows, ...)` takes judged rows.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from config import DEFAULT_BINS
from metrics.frontier import sufficiency_frontier
from pipeline.orchestrator import ResultRow

from ._common import _bin, _rung_metrics, _unique_doc_count, _unique_question_count


def build_table1_headline(
    rows: Sequence[ResultRow],
    *,
    bins: Sequence[str] = DEFAULT_BINS,
    margin_points: float = 3.0,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Build Table 1: bin x ladder headline with frontier and latency."""

    out: list[dict[str, object]] = []
    for bin_name in bins:
        group_rows = [row for row in rows if _bin(row) == bin_name]
        columns, cells, costs = _rung_metrics(group_rows, n_bootstrap=n_bootstrap, seed=seed)
        frontier = sufficiency_frontier(cells, margin_points=margin_points)
        out.append(
            {
                "bin": bin_name,
                "n_questions": _unique_question_count(group_rows),
                "n_docs": _unique_doc_count(group_rows),
                **columns,
                "frontier": frontier,
                "latency_at_frontier_s": costs[frontier].latency_bs1_s if frontier else 0.0,
            }
        )
    return pd.DataFrame(out)
