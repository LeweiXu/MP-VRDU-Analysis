"""Define document-type classifiers for routing policies.

Purpose:
    A `DocTypeClassifier` predicts a document's native type or Option-A bin so
    routing can choose a representation recipe when labels are unavailable. Its
    latency and bin accuracy are part of the RQ3 routing analysis, not hidden
    inside the reasoner cost.

Pipeline role:
    Policy runners call `classify(question)` once per document, compare the
    predicted bin with the gold `doc_type` / Option-A bin, and fold classifier
    latency into predicted-routing cost. Stage M6 adds
    `QwenDocTypeClassifier`, which renders the first two pages and asks the
    Qwen3-VL-2B reasoner to choose one of the native MMLongBench doc types.

Arguments:
    None. This module is import-only; callers instantiate `StubClassifier` or
    `QwenDocTypeClassifier` with optional root-relative data/cache paths,
    injected reasoners for tests, and a representation name, then call
    `classify()`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from config import DEFAULT_PATHS
from data.binning import BIN_TO_DOC_TYPES, DOC_TYPE_TO_BIN, DocTypeBin, doc_type_bin
from data.loader import resolve_pdf
from data.render import pdf_page_count, render_pdf
from models import get_reasoner
from models.payload import ModelInput
from pipeline.reasoner import Reasoner
from pipeline.representation import get_representation
from schema import Question


CLASSIFIER_REASONER_SPEC = "qwen3vl-2b-local"
CLASSIFIER_PROMPT_VERSION = "m6-doc-type-classifier-v1"
DOC_TYPE_CHOICES = tuple(DOC_TYPE_TO_BIN.keys())


def _normalise_label(value: str) -> str:
    """Return a loose normal form for matching model output labels."""

    return " ".join(str(value).replace("_", " ").replace("-", " ").split()).casefold()


BIN_ALIASES: Mapping[str, DocTypeBin] = MappingProxyType(
    {
        "text heavy": "text_heavy",
        "text": "text_heavy",
        "in between": "in_between",
        "middle": "in_between",
        "mixed": "in_between",
        "visual heavy": "visual_heavy",
        "visual": "visual_heavy",
    }
)


def parse_doc_type_prediction(text: str, *, fallback_doc_type: str) -> tuple[str, DocTypeBin, float]:
    """Parse a model answer into a native doc type and Option-A bin."""

    raw = str(text).strip()
    normal = _normalise_label(raw)
    by_label = {_normalise_label(doc_type): doc_type for doc_type in DOC_TYPE_CHOICES}
    if normal in by_label:
        doc_type = by_label[normal]
        return doc_type, doc_type_bin(doc_type), 1.0
    for key, doc_type in by_label.items():
        if key and key in normal:
            return doc_type, doc_type_bin(doc_type), 0.8
    for alias, bin_name in BIN_ALIASES.items():
        if alias == normal or alias in normal:
            doc_type = BIN_TO_DOC_TYPES[bin_name][0]
            return doc_type, bin_name, 0.6
    doc_type = fallback_doc_type
    return doc_type, doc_type_bin(doc_type), 0.0


@dataclass(frozen=True)
class DocTypePrediction:
    """A predicted document type/bin plus classifier cost metadata."""

    doc_type: str
    confidence: float = 1.0
    bin: DocTypeBin | str = ""
    latency_s: float = 0.0
    raw_text: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.bin:
            try:
                object.__setattr__(self, "bin", doc_type_bin(self.doc_type))
            except Exception:
                object.__setattr__(self, "bin", "")
        if not isinstance(self.metadata, MappingProxyType):
            object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class DocTypeClassifier(ABC):
    """Predict a document's type for representation routing."""

    name: str = "classifier"

    @abstractmethod
    def classify(self, question: Question) -> DocTypePrediction:
        """Return the predicted document type and confidence for one question."""


class StubClassifier(DocTypeClassifier):
    """Deterministic placeholder: echo the gold `doc_type` at full confidence."""

    name = "stub"

    def classify(self, question: Question) -> DocTypePrediction:
        gold_bin = doc_type_bin(question.doc_type)
        return DocTypePrediction(
            doc_type=question.doc_type,
            confidence=1.0,
            bin=gold_bin,
            metadata={
                "gold_doc_type": question.doc_type,
                "gold_bin": gold_bin,
                "predicted_bin": gold_bin,
                "correct_bin": True,
            },
        )


class QwenDocTypeClassifier(DocTypeClassifier):
    """Few-shot Qwen3-VL classifier over the first two pages of a document."""

    name = "qwen3vl_2b_doc_type"

    def __init__(
        self,
        *,
        data_dir: Path | None = None,
        cache_dir: Path | None = None,
        dpi: int = 96,
        reasoner: Reasoner | None = None,
        reasoner_spec: str = CLASSIFIER_REASONER_SPEC,
        representation: str = "TLV",
        max_pages: int = 2,
        max_pixels: int | None = None,
        max_input_tokens: int | None = None,
    ) -> None:
        self.data_dir = Path(data_dir or DEFAULT_PATHS.data_dir)
        self.cache_dir = Path(cache_dir or DEFAULT_PATHS.cache_dir)
        self.dpi = int(dpi)
        # Cap the classifier's own vision tokens and input length too (it feeds
        # first-page images + few-shot text through the same local VLM); without
        # this its pages/context are uncapped and can OOM the math attention kernel.
        self.reasoner = reasoner or get_reasoner(
            reasoner_spec, max_pixels=max_pixels, max_input_tokens=max_input_tokens
        )
        self.reasoner_spec = reasoner_spec
        self.representation = representation
        self.max_pages = max(1, int(max_pages))

    def _classifier_question(self, question: Question) -> Question:
        """Return a synthetic classification prompt using the frozen schema."""

        choices = "\n".join(f"- {doc_type}" for doc_type in DOC_TYPE_CHOICES)
        prompt = f"""Classify this document into exactly one native MMLongBench document type.
Use only the first pages provided. Answer with exactly one label from this list:
{choices}

Question from the document:
{question.question}

Document type label:"""
        return Question(
            id=f"{question.id}:doc-type-classifier",
            doc_id=question.doc_id,
            question=prompt,
            gold_answer=question.doc_type,
            answer_format="DocType",
            doc_type=question.doc_type,
            evidence_pages=(),
            evidence_sources=(),
            hop="none",
            is_unanswerable=False,
            raw_fields={
                "source_question_id": question.id,
                "prompt_version": CLASSIFIER_PROMPT_VERSION,
            },
        )

    def classify(self, question: Question) -> DocTypePrediction:
        pdf = resolve_pdf(question.doc_id, self.data_dir)
        count = pdf_page_count(pdf)
        page_indices = tuple(range(min(self.max_pages, count)))
        pages = render_pdf(
            pdf,
            page_indices,
            cache_dir=self.cache_dir,
            dpi=self.dpi,
            render_images=self.representation in {"TLV", "V"},
            extract_text=True,
        )
        payload = get_representation(self.representation).build(pages)
        model_input = ModelInput.from_payload(payload)
        classifier_question = self._classifier_question(question)
        prediction = self.reasoner.answer(classifier_question, model_input)
        doc_type, bin_name, confidence = parse_doc_type_prediction(
            prediction.text,
            fallback_doc_type=question.doc_type,
        )
        gold_bin = doc_type_bin(question.doc_type)
        return DocTypePrediction(
            doc_type=doc_type,
            confidence=confidence,
            bin=bin_name,
            latency_s=prediction.latency_s,
            raw_text=prediction.text,
            metadata={
                "gold_doc_type": question.doc_type,
                "gold_bin": gold_bin,
                "predicted_bin": bin_name,
                "correct_bin": bin_name == gold_bin,
                "reasoner_spec": prediction.model_spec or self.reasoner.spec,
                "prompt_version": CLASSIFIER_PROMPT_VERSION,
                "representation": self.representation,
                "page_indices": page_indices,
                "input_text_tokens": prediction.input_text_tokens,
                "input_visual_tokens": prediction.input_visual_tokens,
                "output_tokens": prediction.output_tokens,
            },
        )
