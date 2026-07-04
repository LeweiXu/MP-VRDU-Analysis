"""Map native MMLongBench document types to the v3 Option-A bins.

Purpose:
    Provides the single source of truth for the deployable doc-type axis used by
    the paper: text-heavy, in-between, and visual-heavy. It also records the
    full-corpus question/document counts used in docs and tests.

Pipeline role:
    Experiment runners, table builders, classifier evaluation, and smoke-corpus
    checks call `doc_type_bin(doc_type)` instead of duplicating bin logic. The
    Section-3 Option-B robustness analysis should replace only this function's
    body behind the same signature.

Arguments:
    None. This is an import-only module; public inputs are native `doc_type`
    strings passed to `canonical_doc_type()` or `doc_type_bin()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping


DocTypeBin = Literal["text_heavy", "in_between", "visual_heavy"]

TEXT_HEAVY_DOC_TYPES = (
    "Administration/Industry file",
    "Academic paper",
    "Research report / Introduction",
)
IN_BETWEEN_DOC_TYPES = (
    "Financial report",
    "Guidebook",
    "Tutorial/Workshop",
)
VISUAL_HEAVY_DOC_TYPES = ("Brochure",)

DEFAULT_BINS: tuple[DocTypeBin, ...] = (
    "text_heavy",
    "in_between",
    "visual_heavy",
)

DOC_TYPE_TO_BIN: Mapping[str, DocTypeBin] = MappingProxyType(
    {
        **{doc_type: "text_heavy" for doc_type in TEXT_HEAVY_DOC_TYPES},
        **{doc_type: "in_between" for doc_type in IN_BETWEEN_DOC_TYPES},
        **{doc_type: "visual_heavy" for doc_type in VISUAL_HEAVY_DOC_TYPES},
    }
)

BIN_TO_DOC_TYPES: Mapping[DocTypeBin, tuple[str, ...]] = MappingProxyType(
    {
        "text_heavy": TEXT_HEAVY_DOC_TYPES,
        "in_between": IN_BETWEEN_DOC_TYPES,
        "visual_heavy": VISUAL_HEAVY_DOC_TYPES,
    }
)


@dataclass(frozen=True)
class BinCount:
    """Full MMLongBench-Doc question/document count for one Option-A bin."""

    questions: int
    documents: int


OPTION_A_BIN_COUNTS: Mapping[DocTypeBin, BinCount] = MappingProxyType(
    {
        "text_heavy": BinCount(questions=578, documents=70),
        "in_between": BinCount(questions=412, documents=50),
        "visual_heavy": BinCount(questions=101, documents=15),
    }
)

_DOC_TYPE_ALIASES = {
    " ".join(doc_type.split()).casefold(): doc_type for doc_type in DOC_TYPE_TO_BIN
}
_DOC_TYPE_ALIASES["research report/introduction"] = "Research report / Introduction"


def canonical_doc_type(doc_type: str) -> str:
    """Return the canonical MMLongBench doc_type label or raise on unknown."""

    collapsed = " ".join(str(doc_type).split())
    canonical = _DOC_TYPE_ALIASES.get(collapsed.casefold())
    if canonical is None:
        valid = ", ".join(sorted(DOC_TYPE_TO_BIN))
        raise ValueError(f"unknown MMLongBench doc_type {doc_type!r}; expected one of: {valid}")
    return canonical


def doc_type_bin(doc_type: str) -> DocTypeBin:
    """Map a native MMLongBench `doc_type` label to its fixed Option-A bin."""

    return DOC_TYPE_TO_BIN[canonical_doc_type(doc_type)]
