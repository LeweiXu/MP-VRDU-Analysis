"""Define the frozen data contracts exchanged across the pipeline.

Purpose:
    Holds the canonical dataclasses for questions, selected pages, rendered page
    content, representation payload parts, model predictions, and judge scores.
    These contracts are the Stage-3 freeze point: later implementations fill in
    behaviour behind them rather than changing their shape.

Pipeline role:
    `Question` comes from `data.loader`; `PageSet` is returned by input
    conditioners; `Page`, `TextPart`, `ImagePart`, and `Payload` are produced by
    rendering and representation composers; `Prediction` and `Score` are emitted
    by reasoners and judges. The modality boundary is enforced here.

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

Arguments:
    None. This module is import-only and exposes dataclasses/helpers.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, Union


Hop = Literal["none", "single", "multi"]
PageSetProvenance = Literal["oracle", "retrieved", "full", "buried"]
Modality = str


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
class TextPart:
    """One ordered text fragment in a payload / model input."""

    text: str


@dataclass(frozen=True)
class ImagePart:
    """One ordered image, referenced by path or carried inline as bytes.

    A part sourced from disk keeps its `image_path`; a part reconstructed from a
    serialised model input (e.g. a base64 data URI) carries decoded `data`
    instead. `read_bytes()`/`data_uri()` hide that difference from callers, so
    the local and API adapters never need to know where the image came from.
    """

    image_path: Path | None = None
    data: bytes | None = None
    mime: str = "image/png"

    def __post_init__(self) -> None:
        if self.image_path is None and self.data is None:
            raise ValueError("ImagePart needs either an image_path or inline data")
        if self.image_path is not None:
            object.__setattr__(self, "image_path", Path(self.image_path))

    def read_bytes(self) -> bytes:
        """Return the raw image bytes regardless of source."""

        if self.data is not None:
            return self.data
        return Path(self.image_path).read_bytes()  # type: ignore[arg-type]

    def data_uri(self) -> str:
        """Return a base64 `data:` URI usable in a chat image_url part."""

        encoded = base64.b64encode(self.read_bytes()).decode("ascii")
        return f"data:{self.mime};base64,{encoded}"


Part = Union[TextPart, ImagePart]


@dataclass(frozen=True)
class Payload:
    """Ordered representation output produced by a `Representation` composer.

    `modality` is the ladder rung that built it (`T`, `TL`, `TLV`, `V`). The
    modality-boundary rule from the spec is enforced structurally here: `T` and
    `TL` payloads may not carry images, only `TLV` and `V` may. The reasoner-
    facing `ModelInput` (`models/payload.py`) is derived from this payload.
    """

    modality: Modality
    parts: tuple[Part, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "parts", tuple(self.parts))
        if "V" not in self.modality and self.image_parts:
            raise ValueError(
                f"modality {self.modality!r} may not attach images "
                "(only representations containing V may carry a visual channel)"
            )

    @property
    def text_parts(self) -> tuple[TextPart, ...]:
        return tuple(p for p in self.parts if isinstance(p, TextPart))

    @property
    def image_parts(self) -> tuple[ImagePart, ...]:
        return tuple(p for p in self.parts if isinstance(p, ImagePart))


@dataclass(frozen=True)
class Prediction:
    """A reasoner's answer plus the cost accounting that Stage 6 fills in.

    Token counts are split text vs visual so the accuracy-cost analysis (RQ4)
    can price the visual channel separately. The Stage 3 stub reasoner returns a
    fixed answer with the cost fields zeroed.
    """

    text: str
    model_spec: str = ""
    input_text_tokens: int = 0
    input_visual_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Score:
    """A judge's verdict on one prediction.

    `value` is the graded correctness in [0, 1]; `correct`/`abstained` are the
    boolean views the accuracy and abstention metrics read. The Stage 3 stub
    judge returns a heuristic; the real different-family judge lands in Stage 7.
    """

    value: float
    correct: bool = False
    abstained: bool = False
    judge_spec: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
