"""Option-A document-type binning for the v3 MP-VRDU study.

This module is the single source of truth for mapping MMLongBench-Doc's native
`doc_type` labels into the three deployable bins used by the paper. Section-3's
data-driven Option B may replace the body behind `doc_type_bin()`, but callers
should keep using this function.
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
