"""Shared data contracts for questions, page sets, payloads, predictions, and scores.

MMLongBench-Doc source mapping for `Question`:

- `doc_id` comes from the benchmark `doc_id` field and resolves to a PDF under
  `.data/mmlongbench/documents/`.
- `question` comes from `question`.
- `gold_answer` comes from `answer`.
- `answer_format` comes from `answer_format`.
- `doc_type` comes from `doc_type`.
- `evidence_pages` comes from `evidence_pages` and is normalised to zero-based
  page indices for PyMuPDF/rendering. The source field is one-based in the
  dataset; the original value remains in `raw_fields`.
- `evidence_sources` comes from `evidence_sources`.
- `hop` is derived from the normalised evidence-page count: `none`, `single`,
  or `multi`.
- `is_unanswerable` is true when the gold answer normalises to
  `"not answerable"`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping


Hop = Literal["none", "single", "multi"]
PageSetProvenance = Literal["oracle", "retrieved", "full", "buried"]


def derive_hop(page_indices: tuple[int, ...]) -> Hop:
    """Return the evidence-hop class implied by normalised evidence pages."""

    if not page_indices:
        return "none"
    if len(page_indices) == 1:
        return "single"
    return "multi"


def is_not_answerable(answer: Any) -> bool:
    """Return whether a gold answer uses the native MMLongBench abstention token."""

    return str(answer).strip().casefold() == "not answerable"


@dataclass(frozen=True)
class Question:
    """Normalised MMLongBench-Doc question used by every later stage."""

    id: str
    doc_id: str
    question: str
    gold_answer: str
    answer_format: str
    doc_type: str
    evidence_pages: tuple[int, ...]
    evidence_sources: tuple[str, ...]
    hop: Hop
    is_unanswerable: bool
    raw_fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_pages", tuple(int(page) for page in self.evidence_pages))
        object.__setattr__(self, "evidence_sources", tuple(str(src) for src in self.evidence_sources))
        object.__setattr__(self, "hop", derive_hop(self.evidence_pages))
        object.__setattr__(self, "is_unanswerable", is_not_answerable(self.gold_answer))
        if not isinstance(self.raw_fields, MappingProxyType):
            object.__setattr__(self, "raw_fields", MappingProxyType(dict(self.raw_fields)))

    @property
    def page_count_required(self) -> int:
        """Minimum page count needed for the gold evidence pages to be valid."""

        return max(self.evidence_pages, default=-1) + 1


@dataclass(frozen=True)
class PageSet:
    """A selected set of zero-based page indices plus provenance."""

    page_indices: tuple[int, ...]
    provenance: PageSetProvenance
    note: str = ""

    def __post_init__(self) -> None:
        unique_sorted = tuple(sorted(dict.fromkeys(int(page) for page in self.page_indices)))
        object.__setattr__(self, "page_indices", unique_sorted)

    @classmethod
    def oracle(cls, question: Question) -> "PageSet":
        """Return a page set containing exactly the gold evidence pages."""

        return cls(question.evidence_pages, "oracle")

    @classmethod
    def full(cls, page_count: int) -> "PageSet":
        """Return a page set covering the whole document."""

        return cls(tuple(range(page_count)), "full")


@dataclass(frozen=True)
class TextSpan:
    """Text extracted from one page, optionally with a PDF-space bounding box."""

    text: str
    bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class Page:
    """Rendered/extracted page payload shared by representations and retrievers."""

    doc_id: str
    index: int
    pdf_path: Path
    image_path: Path | None = None
    text_spans: tuple[TextSpan, ...] = ()

    @property
    def text(self) -> str:
        """Concatenate extracted text spans for text-only consumers."""

        return "\n".join(span.text for span in self.text_spans if span.text)


@dataclass(frozen=True)
class Payload:
    """Placeholder representation payload filled by later pipeline stages."""

    parts: tuple[Any, ...] = ()


@dataclass(frozen=True)
class Prediction:
    """Placeholder model prediction filled by later reasoner stages."""

    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Score:
    """Placeholder judge score filled by later metric stages."""

    value: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
