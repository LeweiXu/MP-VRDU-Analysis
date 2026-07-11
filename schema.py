"""Frozen data contracts (Question, PageSet, Page, Payload, Prediction, Score,
PredictionRow, ResultRow) and the per-cell telemetry every run records for one cell."""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, Union


Hop = Literal["none", "single", "multi"]
PageSetProvenance = Literal["oracle", "retrieved", "full", "similarity"]
Modality = str

# The three outcomes a cell can record. Every cell writes exactly one row with
# one of these, so a failed cell is data (a row with status oom/error), never a
# hole in the jsonl.
STATUS_VALUES: tuple[str, ...] = ("ok", "oom", "error")


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


def tokens_dropped(total_text_tokens: int, text_tokens_fed: int) -> int:
    """Text tokens that never reached the reasoner. Should always read zero.

    There is no input-token cap, so the reasoner is fed the whole text sequence
    and this is a canary: a nonzero value means something truncated when nothing
    should have, i.e. a bug, not an analysis signal.
    """

    return int(total_text_tokens) - int(text_tokens_fed)


def truncation_occurred(total_text_tokens: int, text_tokens_fed: int) -> bool:
    """True when any text token was dropped. Expected False on every cell."""

    return tokens_dropped(total_text_tokens, text_tokens_fed) > 0


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
    # Manual-annotation modality bin (text-dominant / mixed-modality /
    # visual-dominant) and scan status, stamped from the annotation table by
    # data.binning. Empty until that join runs.
    bin_label: str = ""
    scan_label: str = ""
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
    """Text extracted from one page."""

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
    modality-boundary rule is enforced structurally: `T` and `TL` payloads may
    not carry images, only `TLV` and `V` may. The reasoner-facing `ModelInput`
    (`models/payload.py`) is derived from this payload.
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
    """A reasoner's answer plus the per-cell cost accounting.

    Token counts are split text vs visual so the cost analysis can price the
    visual channel separately. Latency is split into prefill (ingesting the
    representation, the cost the ladder changes) and decode.
    """

    text: str
    model_spec: str = ""
    total_text_tokens: int = 0
    total_visual_tokens: int = 0
    text_tokens_fed: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    prefill_latency_s: float = 0.0
    decode_latency_s: float = 0.0
    peak_vram_bytes: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Score:
    """A judge's verdict on one prediction.

    `value` is the graded correctness in [0, 1]; `correct`/`abstained` are the
    boolean views the accuracy and abstention scorers read.
    """

    value: float
    correct: bool = False
    abstained: bool = False
    judge_spec: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionRow:
    """One cell's generation output plus per-cell telemetry, before any judging.

    This is what the generate phase writes to `predictions.jsonl`, one per cell
    regardless of outcome (a failed cell still writes a row with `status` in
    {oom, error} and a `skipped_reason`). It is exactly a `ResultRow` minus the
    judge-derived fields (`result_key`, `judge_spec`, `score`, `correct`,
    `abstained`); the judge phase adds those to build the `ResultRow`, so
    `results.jsonl` is a strict superset of `predictions.jsonl`.
    """

    prediction_key: str
    # identity / provenance
    question_id: str
    doc_id: str
    doc_type: str
    bin_label: str
    scan_label: str
    hop: str
    is_unanswerable: bool
    evidence_sources: tuple[str, ...]
    condition: str
    provenance: str
    page_indices: tuple[int, ...]
    representation: str
    model_spec: str
    machine: str
    status: str
    skipped_reason: str
    oom_occurred: bool
    # answer (no judge verdict yet)
    answer: str
    # tokens (cap removed: fed must equal total; dropped is the zero-canary)
    total_text_tokens: int
    total_visual_tokens: int
    text_tokens_fed: int
    output_tokens: int
    tokens_dropped: int
    truncation_occurred: bool
    # latency split
    latency_s: float
    prefill_latency_s: float
    decode_latency_s: float
    # memory
    peak_vram_bytes: int
    # the visual-resolution preset this cell was fed (part of the cell key)
    visual_resolution: str = ""
    note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        data = asdict(self)
        data["page_indices"] = list(self.page_indices)
        data["evidence_sources"] = list(self.evidence_sources)
        return json.dumps(data, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PredictionRow":
        data = dict(data)
        data["page_indices"] = tuple(data.get("page_indices", ()))
        data["evidence_sources"] = tuple(data.get("evidence_sources", ()))
        return cls(**data)

    def as_prediction(self) -> "Prediction":
        """Rebuild the `Prediction` this row's answer + telemetry came from.

        Lets a judge (or a prediction-cache hit) reuse the reasoner output without
        re-running the model; `answer` maps back to `Prediction.text`.
        """

        return Prediction(
            text=self.answer,
            model_spec=self.model_spec,
            total_text_tokens=self.total_text_tokens,
            total_visual_tokens=self.total_visual_tokens,
            text_tokens_fed=self.text_tokens_fed,
            output_tokens=self.output_tokens,
            latency_s=self.latency_s,
            prefill_latency_s=self.prefill_latency_s,
            decode_latency_s=self.decode_latency_s,
            peak_vram_bytes=self.peak_vram_bytes,
        )

    def to_result_row(self, score: "Score", result_key: str) -> "ResultRow":
        """Add the judge verdict to build the full `ResultRow` for a cell."""

        return ResultRow(
            result_key=result_key,
            prediction_key=self.prediction_key,
            question_id=self.question_id,
            doc_id=self.doc_id,
            doc_type=self.doc_type,
            bin_label=self.bin_label,
            scan_label=self.scan_label,
            hop=self.hop,
            is_unanswerable=self.is_unanswerable,
            evidence_sources=self.evidence_sources,
            condition=self.condition,
            provenance=self.provenance,
            page_indices=self.page_indices,
            representation=self.representation,
            model_spec=self.model_spec,
            judge_spec=score.judge_spec,
            machine=self.machine,
            status=self.status,
            skipped_reason=self.skipped_reason,
            oom_occurred=self.oom_occurred,
            answer=self.answer,
            score=score.value,
            correct=score.correct,
            abstained=score.abstained,
            total_text_tokens=self.total_text_tokens,
            total_visual_tokens=self.total_visual_tokens,
            text_tokens_fed=self.text_tokens_fed,
            output_tokens=self.output_tokens,
            tokens_dropped=self.tokens_dropped,
            truncation_occurred=self.truncation_occurred,
            latency_s=self.latency_s,
            prefill_latency_s=self.prefill_latency_s,
            decode_latency_s=self.decode_latency_s,
            peak_vram_bytes=self.peak_vram_bytes,
            visual_resolution=self.visual_resolution,
            note=self.note,
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class ResultRow:
    """One cell's fully-scored result plus the uniform per-cell telemetry.

    Every cell emits exactly one of these regardless of outcome; a failed cell
    still writes a row with `status` in {oom, error} and a `skipped_reason`. The
    truncation fields (`tokens_dropped`, `truncation_occurred`) are a canary that
    should read zero now that there is no input-token cap.
    """

    # keys
    result_key: str
    prediction_key: str
    # identity / provenance
    question_id: str
    doc_id: str
    doc_type: str
    bin_label: str
    scan_label: str
    hop: str
    is_unanswerable: bool
    evidence_sources: tuple[str, ...]
    condition: str
    provenance: str
    page_indices: tuple[int, ...]
    representation: str
    model_spec: str
    judge_spec: str
    machine: str
    status: str
    skipped_reason: str
    oom_occurred: bool
    # answer + judge verdict
    answer: str
    score: float
    correct: bool
    abstained: bool
    # tokens (cap removed: fed must equal total; dropped is the zero-canary)
    total_text_tokens: int
    total_visual_tokens: int
    text_tokens_fed: int
    output_tokens: int
    tokens_dropped: int
    truncation_occurred: bool
    # latency split
    latency_s: float
    prefill_latency_s: float
    decode_latency_s: float
    # memory
    peak_vram_bytes: int
    # the visual-resolution preset this cell was fed (part of the cell key)
    visual_resolution: str = ""
    note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        data = asdict(self)
        data["page_indices"] = list(self.page_indices)
        data["evidence_sources"] = list(self.evidence_sources)
        return json.dumps(data, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResultRow":
        data = dict(data)
        data["page_indices"] = tuple(data.get("page_indices", ()))
        data["evidence_sources"] = tuple(data.get("evidence_sources", ()))
        return cls(**data)
