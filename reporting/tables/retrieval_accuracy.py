"""Page-retrieval accuracy benchmark: precision/recall/F1 per method, both broken
out by doc_type and aggregated over all docs, from the retrieval side-artifact
(covers methods never fed to the reasoner)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._common import Table, doc_type_of, group_by


def _mean(values: Sequence[float]) -> str:
    return f"{sum(values) / len(values):.3f}" if values else "-"


def _metrics(group: Sequence[Any]) -> list[str]:
    """The P/R/F1/n cells for one group of retrieval rows."""

    return [
        _mean([float(getattr(r, "precision", 0.0)) for r in group]),
        _mean([float(getattr(r, "recall", 0.0)) for r in group]),
        _mean([float(getattr(r, "f1", 0.0)) for r in group]),
        str(len(group)),
    ]


def build(retrieval_rows: Sequence[Any]) -> Table:
    """One row per (retriever, k, doc_type): macro P/R/F1 over its questions."""

    columns = ["retriever", "modality", "k", "doc_type", "P", "R", "F1", "n"]
    by_group = group_by(
        retrieval_rows,
        lambda r: (getattr(r, "retriever", ""), getattr(r, "modality", ""), int(getattr(r, "k", 0)), doc_type_of(r)),
    )
    table_rows: list[list[str]] = []
    for (retriever, modality, k, dt) in sorted(by_group, key=lambda t: (t[0], t[1], t[2], t[3])):
        table_rows.append([retriever, modality, str(k), dt, *_metrics(by_group[(retriever, modality, k, dt)])])
    return Table(
        key="retrieval_accuracy",
        title="Retrieval accuracy: page P/R/F1 by method and doc_type",
        columns=columns,
        rows=table_rows,
    )


def summary(retrieval_rows: Sequence[Any]) -> Table:
    """Best-F1 operating point per retriever/modality (the k with the highest mean F1)."""

    columns = ["retriever", "modality", "best_k", "P", "R", "F1", "n"]
    by_group = group_by(
        retrieval_rows,
        lambda r: (getattr(r, "retriever", ""), getattr(r, "modality", ""), int(getattr(r, "k", 0))),
    )
    best: dict[tuple[str, str], tuple[float, int, list[Any]]] = {}
    for (retriever, modality, k), group in by_group.items():
        f1s = [float(getattr(r, "f1", 0.0)) for r in group]
        mean_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        key = (retriever, modality)
        if key not in best or mean_f1 > best[key][0]:
            best[key] = (mean_f1, k, list(group))
    table_rows = [
        [retriever, modality, str(best[(retriever, modality)][1]), *_metrics(best[(retriever, modality)][2])]
        for (retriever, modality) in sorted(best)
    ]
    return Table(key="retrieval_accuracy_summary",
                 title="Retrieval accuracy (summary): best-F1 operating point per method",
                 columns=columns, rows=table_rows,
                 note="best_k = the depth k with the highest mean F1 for that method (all doc_types).")


def build_overall(retrieval_rows: Sequence[Any]) -> Table:
    """One row per (retriever, k): macro P/R/F1 over all questions, no doc_type split."""

    columns = ["retriever", "modality", "k", "P", "R", "F1", "n"]
    by_group = group_by(
        retrieval_rows,
        lambda r: (getattr(r, "retriever", ""), getattr(r, "modality", ""), int(getattr(r, "k", 0))),
    )
    table_rows: list[list[str]] = []
    for (retriever, modality, k) in sorted(by_group, key=lambda t: (t[0], t[1], t[2])):
        table_rows.append([retriever, modality, str(k), *_metrics(by_group[(retriever, modality, k)])])
    return Table(
        key="retrieval_accuracy_overall",
        title="Retrieval accuracy: page P/R/F1 by method (all doc_types)",
        columns=columns,
        rows=table_rows,
    )


def build_by_dpi(retrieval_rows: Sequence[Any]) -> Table:
    """One row per (retriever, k, dpi): the render-resolution sweep for visual retrieval.

    Reads the same rows as the other retrieval tables; a run stamps each row with its
    render `dpi`, so pointing the build at a merged multi-dpi `retrieval.jsonl` yields
    the P/R/F1-vs-dpi comparison. Single-dpi builds show one dpi per method.
    """

    columns = ["retriever", "modality", "k", "dpi", "P", "R", "F1", "n"]
    by_group = group_by(
        retrieval_rows,
        lambda r: (getattr(r, "retriever", ""), getattr(r, "modality", ""),
                   int(getattr(r, "k", 0)), int(getattr(r, "dpi", 0))),
    )
    table_rows: list[list[str]] = []
    for (retriever, modality, k, dpi) in sorted(by_group, key=lambda t: (t[0], t[1], t[2], t[3])):
        table_rows.append([retriever, modality, str(k), str(dpi), *_metrics(by_group[(retriever, modality, k, dpi)])])
    return Table(
        key="retrieval_dpi",
        title="Retrieval accuracy: page P/R/F1 by method and render DPI",
        columns=columns,
        rows=table_rows,
    )
