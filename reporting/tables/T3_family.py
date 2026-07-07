"""Table 3: cross-family replication (does the frontier hold on another model?).

Purpose:
    Runs the same per-bin frontier over every model spec present in the rows (G1's
    Qwen3-VL-8B plus G2's InternVL3-8B) and marks whether each family's per-bin
    frontier matches the primary (Qwen3-VL-8B). Table 8 reuses this builder for the
    scale series.

Arguments:
    None. Import-only; `build_table3_family_replication(rows, ...)` takes rows.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from config import DEFAULT_BINS
from metrics.frontier import sufficiency_frontier
from pipeline.orchestrator import ResultRow

from ._common import _rung_metrics, _safe_bin, _size_label, _unique_question_count


def build_table3_family_replication(
    rows: Sequence[ResultRow],
    *,
    bins: Sequence[str] = DEFAULT_BINS,
    margin_points: float = 3.0,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Build Table 3: model-family replication with primary-frontier match."""

    out: list[dict[str, object]] = []
    model_specs = sorted({row.model_spec for row in rows if _safe_bin(row)}) or [""]
    primary_spec = next((spec for spec in model_specs if spec.startswith("qwen3vl-8b")), model_specs[0])
    primary_frontiers: dict[str, str] = {}
    for bin_name in bins:
        primary_rows = [row for row in rows if row.model_spec == primary_spec and _safe_bin(row) == bin_name]
        _, cells, _ = _rung_metrics(primary_rows, n_bootstrap=n_bootstrap, seed=seed)
        primary_frontiers[bin_name] = sufficiency_frontier(cells, margin_points=margin_points)
    for model_spec in model_specs:
        model_rows = [row for row in rows if row.model_spec == model_spec]
        for bin_name in bins:
            group_rows = [row for row in model_rows if _safe_bin(row) == bin_name]
            columns, cells, _ = _rung_metrics(group_rows, n_bootstrap=n_bootstrap, seed=seed)
            frontier = sufficiency_frontier(cells, margin_points=margin_points)
            out.append(
                {
                    "model_spec": model_spec,
                    "model_size": _size_label(model_spec),
                    "bin": bin_name,
                    "n_questions": _unique_question_count(group_rows),
                    **columns,
                    "frontier": frontier,
                    "primary_model_spec": primary_spec,
                    "primary_frontier": primary_frontiers.get(bin_name, ""),
                    "matches_primary_frontier": bool(frontier and frontier == primary_frontiers.get(bin_name, "")),
                }
            )
    return pd.DataFrame(out)
