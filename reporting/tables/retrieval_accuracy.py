"""Page-retrieval accuracy benchmark: precision/recall/F1 per method and bin,
from the retrieval side-artifact (covers methods never fed to the reasoner)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._common import Table, bin_of, group_by


def _mean(values: Sequence[float]) -> str:
    return f"{sum(values) / len(values):.3f}" if values else "-"


def build(retrieval_rows: Sequence[Any]) -> Table:
    """One row per (retriever, k, bin): macro P/R/F1 over its questions."""

    columns = ["retriever", "modality", "k", "bin", "P", "R", "F1", "n"]
    by_group = group_by(
        retrieval_rows,
        lambda r: (getattr(r, "retriever", ""), getattr(r, "modality", ""), int(getattr(r, "k", 0)), bin_of(r)),
    )
    table_rows: list[list[str]] = []
    for (retriever, modality, k, b) in sorted(by_group, key=lambda t: (t[0], t[1], t[2], t[3])):
        group = by_group[(retriever, modality, k, b)]
        table_rows.append(
            [
                retriever,
                modality,
                str(k),
                b,
                _mean([float(getattr(r, "precision", 0.0)) for r in group]),
                _mean([float(getattr(r, "recall", 0.0)) for r in group]),
                _mean([float(getattr(r, "f1", 0.0)) for r in group]),
                str(len(group)),
            ]
        )
    return Table(
        key="retrieval_accuracy",
        title="Retrieval accuracy: page P/R/F1 by method and bin",
        columns=columns,
        rows=table_rows,
    )
