"""Table 4: dataset replication on the held-out MMLongBench subset.

Purpose:
    Same per-bin frontier as Table 1, but over a disjoint set of MMLongBench
    documents (text_heavy / in_between) plus the reused visual_heavy questions (see
    `experiments/corpus.py::sample_table4_replication`). Compare its frontier
    column against Table 1 to judge whether the recipe replicates on unseen docs.

Arguments:
    None. Import-only; `build_table4_dataset_replication(rows, ...)` takes rows.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from config import DEFAULT_BINS
from metrics.frontier import sufficiency_frontier
from pipeline.orchestrator import ResultRow

from ._common import _rung_metrics, _safe_bin, _unique_doc_count, _unique_question_count


def build_table4_dataset_replication(
    rows: Sequence[ResultRow],
    *,
    bins: Sequence[str] = DEFAULT_BINS,
    margin_points: float = 3.0,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Build Table 4: RQ1 replication on the held-out MMLongBench subset, per domain."""

    out: list[dict[str, object]] = []
    for bin_name in bins:
        group_rows = [row for row in rows if _safe_bin(row) == bin_name]
        columns, cells, costs = _rung_metrics(group_rows, n_bootstrap=n_bootstrap, seed=seed)
        frontier = sufficiency_frontier(cells, margin_points=margin_points)
        out.append(
            {
                "dataset": "mmlongbench_heldout",
                "bin": bin_name,
                "n_questions": _unique_question_count(group_rows),
                "n_docs": _unique_doc_count(group_rows),
                **columns,
                "frontier": frontier,
                "latency_at_frontier_s": costs[frontier].latency_bs1_s if frontier else 0.0,
            }
        )
    return pd.DataFrame(out)
