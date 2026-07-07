"""Table 2: analytical breakdown by question type (not for deployment).

Purpose:
    Re-slices G1's oracle rows by the analytical question-type bucket (single-hop
    text / table / chart-figure / multi-hop) within each bin. This explains the
    recipe (mechanism); it is never used in a deployment recommendation, because a
    practitioner does not know a question's type in advance.

Arguments:
    None. Import-only; `build_table2_analytical(rows, ...)` takes judged rows.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from config import DEFAULT_BINS
from pipeline.orchestrator import ResultRow

from ._common import (
    QUESTION_TYPES,
    _rung_metrics,
    _safe_bin,
    _unique_doc_count,
    _unique_question_count,
    analytical_question_type,
)


def build_table2_analytical(
    rows: Sequence[ResultRow],
    *,
    bins: Sequence[str] = DEFAULT_BINS,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Build Table 2: bin x analytical question-type slice."""

    out: list[dict[str, object]] = []
    for bin_name in bins:
        for question_type in QUESTION_TYPES:
            group_rows = [
                row
                for row in rows
                if _safe_bin(row) == bin_name and analytical_question_type(row) == question_type
            ]
            columns, _, _ = _rung_metrics(group_rows, n_bootstrap=n_bootstrap, seed=seed)
            out.append(
                {
                    "bin": bin_name,
                    "question_type": question_type,
                    "n_questions": _unique_question_count(group_rows),
                    "n_docs": _unique_doc_count(group_rows),
                    **columns,
                }
            )
    return pd.DataFrame(out)
