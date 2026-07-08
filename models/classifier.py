"""Document-type / bin classifier used for routing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from config import DEFAULT_PATHS
from data.binning import BINS
from data.loader import resolve_pdf
from data.render import pdf_page_count, render_pdf
from models import get_reasoner
from models.payload import ModelInput
from pipeline.reasoner import Reasoner
from pipeline.representation import get_representation
from schema import Question


CLASSIFIER_REASONER_SPEC = "qwen3vl-2b-local"
CLASSIFIER_PROMPT_VERSION = "bin-classifier-v1"

# Loose surface forms that map onto the three bins.
BIN_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "text dominant": "text-dominant",
        "text": "text-dominant",
        "mixed modality": "mixed-modality",
        "mixed": "mixed-modality",
        "in between": "mixed-modality",
        "visual dominant": "visual-dominant",
        "visual": "visual-dominant",
    }
)


def _normalise_label(value: str) -> str:
    """Return a loose normal form for matching model output labels."""

    return " ".join(str(value).replace("_", " ").replace("-", " ").split()).casefold()


def parse_bin_prediction(text: str, *, fallback_bin: str = "") -> tuple[str, float]:
    """Parse a model answer into one of the three modality bins with a confidence."""

    normal = _normalise_label(text)
    by_label = {_normalise_label(b): b for b in BINS}
    if normal in by_label:
        return by_label[normal], 1.0
    for key, bin_label in by_label.items():
        if key and key in normal:
            return bin_label, 0.8
    for alias, bin_label in BIN_ALIASES.items():
        if alias == normal or alias in normal:
            return bin_label, 0.6
    return fallback_bin, 0.0


@dataclass(frozen=True)
class BinPrediction:
    """A predicted modality bin plus classifier cost metadata."""

    bin: str
    confidence: float = 1.0
    latency_s: float = 0.0
    raw_text: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, MappingProxyType):
            object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class BinClassifier(ABC):
    """Predict a document's modality bin for representation routing."""

    name: str = "classifier"

    @abstractmethod
    def classify(self, question: Question) -> BinPrediction:
        """Return the predicted modality bin and confidence for one question."""


class StubClassifier(BinClassifier):
    """Deterministic placeholder: echo the annotated `bin_label` at full confidence."""

    name = "stub"

    def classify(self, question: Question) -> BinPrediction:
        gold = question.bin_label
        return BinPrediction(
            bin=gold,
            confidence=1.0,
            metadata={"gold_bin": gold, "predicted_bin": gold, "correct_bin": True},
        )


class QwenBinClassifier(BinClassifier):
    """Few-shot Qwen3-VL classifier over the first pages of a document."""

    name = "qwen3vl_2b_bin"

    def __init__(
        self,
        *,
        data_dir: Path | None = None,
        cache_dir: Path | None = None,
        dpi: int = 96,
        reasoner: Reasoner | None = None,
        reasoner_spec: str = CLASSIFIER_REASONER_SPEC,
        representation: str = "V",
        max_pages: int = 2,
        max_pixels: int | None = None,
    ) -> None:
        self.data_dir = Path(data_dir or DEFAULT_PATHS.data_dir)
        self.cache_dir = Path(cache_dir or DEFAULT_PATHS.cache_dir)
        self.dpi = int(dpi)
        self.reasoner = reasoner or get_reasoner(reasoner_spec, max_pixels=max_pixels)
        self.reasoner_spec = reasoner_spec
        self.representation = representation
        self.max_pages = max(1, int(max_pages))

    def _classifier_question(self, question: Question) -> Question:
        """Return a synthetic classification prompt using the frozen schema."""

        choices = "\n".join(f"- {b}" for b in BINS)
        prompt = f"""Classify this document by which modality dominates its information content.
Use only the first pages provided. Answer with exactly one label from this list:
{choices}

Modality label:"""
        return Question(
            id=f"{question.id}:bin-classifier",
            doc_id=question.doc_id,
            question=prompt,
            gold_answer=question.bin_label,
            answer_format="ModalityBin",
            doc_type=question.doc_type,
            evidence_pages=(),
            evidence_sources=(),
            hop="none",
            is_unanswerable=False,
            raw_fields={"source_question_id": question.id, "prompt_version": CLASSIFIER_PROMPT_VERSION},
        )

    def classify(self, question: Question) -> BinPrediction:
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
        prediction = self.reasoner.answer(self._classifier_question(question), model_input)
        predicted, confidence = parse_bin_prediction(prediction.text, fallback_bin=question.bin_label)
        gold = question.bin_label
        return BinPrediction(
            bin=predicted,
            confidence=confidence,
            latency_s=prediction.latency_s,
            raw_text=prediction.text,
            metadata={
                "gold_bin": gold,
                "predicted_bin": predicted,
                "correct_bin": predicted == gold,
                "reasoner_spec": prediction.model_spec or self.reasoner.spec,
                "prompt_version": CLASSIFIER_PROMPT_VERSION,
                "representation": self.representation,
                "page_indices": page_indices,
                "peak_vram_bytes": prediction.peak_vram_bytes,
            },
        )
