"""Looks up a document's bin_label from the manual annotation table and stamps
it onto questions."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Mapping

from data.annotations import BIN_LABELS, DocLabel, load_annotations
from schema import Question

# Bins ordered text -> visual, the shared display/iteration order.
BINS: tuple[str, ...] = BIN_LABELS


def bin_for_doc(doc_id: str, table: Mapping[str, DocLabel] | None = None) -> str:
    """Return a document's bin_label, or "" when it is not labelled yet."""

    labels = table if table is not None else load_annotations()
    label = labels.get(doc_id)
    return label.bin_label if label else ""


def scan_for_doc(doc_id: str, table: Mapping[str, DocLabel] | None = None) -> str:
    """Return a document's scan_label, or "" when it is not labelled yet."""

    labels = table if table is not None else load_annotations()
    label = labels.get(doc_id)
    return label.scan_label if label else ""


def stamp_bins(
    questions: Sequence[Question],
    table: Mapping[str, DocLabel] | None = None,
    *,
    require_complete: bool = False,
) -> list[Question]:
    """Return copies of the questions with bin_label/scan_label filled from the table.

    The manual annotation table is an optional enrichment: `doc_type` is left
    untouched (a cell always keeps its native document type), and any document the
    table does not cover just keeps blank bin/scan labels, which reporting buckets
    as `(unlabeled)`. So a partial or absent sheet still runs. Opt in to
    `require_complete` to instead raise when the sheet exists but misses some docs
    (useful once you intend the annotation to be final).
    """

    labels = table if table is not None else load_annotations()
    if labels and require_complete:
        missing = sorted({q.doc_id for q in questions if q.doc_id not in labels})
        if missing:
            raise ValueError(
                f"annotation table covers {len(labels)} docs but {len(missing)} corpus documents are "
                f"unlabelled (e.g. {missing[:3]}); complete annotations/doc_labels.csv before running"
            )
    stamped: list[Question] = []
    for question in questions:
        label = labels.get(question.doc_id)
        if label is None:
            stamped.append(question)
            continue
        stamped.append(
            dataclasses.replace(question, bin_label=label.bin_label, scan_label=label.scan_label)
        )
    return stamped
