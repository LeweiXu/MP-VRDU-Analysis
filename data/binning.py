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


def stamp_bins(questions: Sequence[Question], table: Mapping[str, DocLabel] | None = None) -> list[Question]:
    """Return copies of the questions with bin_label/scan_label filled from the table.

    A document missing from the table keeps blank labels, so an unlabelled corpus
    still loads (the labels just do not drive any binning until filled).
    """

    labels = table if table is not None else load_annotations()
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
