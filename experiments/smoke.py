"""Frozen MMLongBench-Doc smoke corpus for the v3 MVP stages.

The smoke set is selected at the document level and then includes all questions
for those documents. It covers all seven native `doc_type` labels and therefore
all three Option-A bins while keeping documents short enough for quick parser and
GPU integration checks.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from data.binning import DocTypeBin, doc_type_bin
from data.loader import load_mmlongbench
from schema import Question


@dataclass(frozen=True)
class SmokeDocument:
    """One frozen smoke-corpus document and its native MMLongBench doc type."""

    doc_id: str
    doc_type: str

    @property
    def bin(self) -> DocTypeBin:
        return doc_type_bin(self.doc_type)


SMOKE_DOCUMENTS: tuple[SmokeDocument, ...] = (
    SmokeDocument("2303.05039v2.pdf", "Academic paper"),
    SmokeDocument("7c3f6204b3241f142f0f8eb8e1fefe7a.pdf", "Administration/Industry file"),
    SmokeDocument("BRO-GL-MMONEY.pdf", "Brochure"),
    SmokeDocument("f86d073b0d735ac873a65d906ba82758.pdf", "Financial report"),
    SmokeDocument("8dfc21ec151fb9d3578fc32d5c4e5df9.pdf", "Guidebook"),
    SmokeDocument("379f44022bb27aa53efd5d322c7b57bf.pdf", "Research report / Introduction"),
    SmokeDocument("0e94b4197b10096b1f4c699701570fbf.pdf", "Tutorial/Workshop"),
)

SMOKE_DOC_IDS: tuple[str, ...] = tuple(doc.doc_id for doc in SMOKE_DOCUMENTS)
SMOKE_DOC_TYPES: Mapping[str, str] = MappingProxyType(
    {doc.doc_id: doc.doc_type for doc in SMOKE_DOCUMENTS}
)
_SMOKE_DOC_ORDER = {doc_id: index for index, doc_id in enumerate(SMOKE_DOC_IDS)}


def smoke_doc_bins() -> Mapping[str, DocTypeBin]:
    """Return the frozen smoke doc_id -> Option-A bin mapping."""

    return MappingProxyType({doc.doc_id: doc.bin for doc in SMOKE_DOCUMENTS})


def select_smoke_questions(
    questions: Iterable[Question],
    *,
    require_all_docs: bool = False,
) -> list[Question]:
    """Return all questions whose `doc_id` belongs to the frozen smoke corpus."""

    selected: list[Question] = []
    seen_doc_ids: set[str] = set()
    for question in questions:
        expected_doc_type = SMOKE_DOC_TYPES.get(question.doc_id)
        if expected_doc_type is None:
            continue
        if question.doc_type != expected_doc_type:
            raise ValueError(
                f"smoke doc {question.doc_id!r} has doc_type {question.doc_type!r}; "
                f"expected {expected_doc_type!r}"
            )
        selected.append(question)
        seen_doc_ids.add(question.doc_id)

    if require_all_docs:
        missing = tuple(doc_id for doc_id in SMOKE_DOC_IDS if doc_id not in seen_doc_ids)
        if missing:
            raise ValueError(f"smoke corpus is missing documents: {', '.join(missing)}")

    selected.sort(key=lambda q: (_SMOKE_DOC_ORDER[q.doc_id], q.id))
    return selected


def load_smoke_questions(data_dir: Path | None = None) -> list[Question]:
    """Load the staged MMLongBench-Doc corpus and filter it to smoke questions."""

    return select_smoke_questions(load_mmlongbench(data_dir=data_dir), require_all_docs=True)
