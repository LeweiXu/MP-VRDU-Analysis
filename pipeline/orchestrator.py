"""Composes the reasoner stages of one cell, owns the jsonl cell caches (predictions
written by generate, results by the judge phase), and captures per-cell telemetry."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path

from config import DEFAULT_PROMPT_MODE, PROMPT_MODES, ExperimentConfig
from data.loader import resolve_pdf
from data.render import pdf_page_count, render_pdf
from experiments.engine.paths import prediction_key as make_prediction_key
from models import get_reasoner
from models.payload import ModelInput
from pipeline.conditioner import InputConditioner
from pipeline.reasoner import Reasoner
from pipeline.representation import Representation, get_representation
from schema import Page, PageSet, Prediction, PredictionRow, Question, ResultRow, tokens_dropped, truncation_occurred

log = logging.getLogger("mpvrdu.orchestrator")


def current_machine() -> str:
    """Provenance label for the box completing a row (drives nothing)."""

    return os.environ.get("MPVRDU_MACHINE", "local")


class ResultCache:
    """Append-only jsonl cache of scored rows keyed by result_key."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._index: dict[str, ResultRow] = {}
        self._handle = None
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                row = ResultRow.from_dict(json.loads(line))
                self._index[row.result_key] = row

    def get(self, result_key: str) -> ResultRow | None:
        return self._index.get(result_key)

    def put(self, row: ResultRow) -> None:
        if row.result_key in self._index:
            return
        self._index[row.result_key] = row
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a")
        self._handle.write(row.to_json() + "\n")
        self._handle.flush()

    def __iter__(self) -> Iterator[ResultRow]:
        return iter(self._index.values())

    def __len__(self) -> int:
        return len(self._index)


class PredictionCache:
    """Append-only jsonl cache of `PredictionRow`s keyed by prediction_key (no judge).

    This is what the generate phase writes to `predictions.jsonl`: one row per cell
    including failures, carrying everything a `ResultRow` needs except the judge
    verdict, so the judge phase can rebuild the row without re-running the model.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._index: dict[str, PredictionRow] = {}
        self._handle = None
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                record = PredictionRow.from_dict(json.loads(line))
                self._index[record.prediction_key] = record

    def get(self, prediction_key: str) -> PredictionRow | None:
        return self._index.get(prediction_key)

    def put(self, record: PredictionRow) -> None:
        if record.prediction_key in self._index:
            return
        self._index[record.prediction_key] = record
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a")
        self._handle.write(record.to_json() + "\n")
        self._handle.flush()

    def __iter__(self) -> Iterator[PredictionRow]:
        return iter(self._index.values())

    def __len__(self) -> int:
        return len(self._index)


class Orchestrator:
    """Compose the reasoner path for one cell, writing an unjudged `PredictionRow`."""

    def __init__(
        self,
        config: ExperimentConfig,
        reasoner: Reasoner | None = None,
        prediction_cache: PredictionCache | None = None,
        machine: str | None = None,
        visual_resolution: str | None = None,
    ) -> None:
        self.config = config
        self.reasoner = reasoner if reasoner is not None else get_reasoner(config.reasoner_spec)
        self.prediction_cache = prediction_cache
        self.machine = machine or current_machine()
        # The resolution this orchestrator feeds every cell; part of the cell key.
        self.visual_resolution = visual_resolution or config.visual_resolution
        self._page_count_cache: dict[str, int] = {}

    # -- page resolution --------------------------------------------------

    def page_count(self, question: Question) -> int:
        """Total page count for a question's document (cached per doc)."""

        if question.doc_id not in self._page_count_cache:
            pdf = resolve_pdf(question.doc_id, self.config.paths.data_dir)
            self._page_count_cache[question.doc_id] = pdf_page_count(pdf)
        return self._page_count_cache[question.doc_id]

    def render_pages(self, question: Question, page_set: PageSet) -> list[Page]:
        """Render the selected pages for a question."""

        if not page_set.page_indices:
            return []
        pdf = resolve_pdf(question.doc_id, self.config.paths.data_dir)
        return render_pdf(pdf, page_set.page_indices, cache_dir=self.config.paths.cache_dir, dpi=self.config.dpi)

    # -- keys -------------------------------------------------------------

    def _prediction_key(self, question: Question, conditioner: InputConditioner, modality: str,
                        page_indices: tuple[int, ...]) -> str:
        return make_prediction_key(
            question.id, question.doc_id, conditioner.name, modality, self.reasoner.spec,
            page_indices, self.visual_resolution,
        )

    # -- the run loop -----------------------------------------------------

    def prewarm_cell(
        self,
        question: Question,
        conditioner: InputConditioner,
        representation: Representation | str,
    ) -> PageSet:
        """Warm the retrieval and render caches without loading the reasoner.

        Conditioning runs the retriever (warming its cache) and rendering writes
        the page PNGs, so the later `run_cell` needs neither once the caches are
        warm. Returns the conditioned `PageSet` so the caller can reuse it (e.g.
        to warm parser markdown) without conditioning the cell a second time.
        Parser markdown is warmed separately in its own env, so this does not
        build TL/TLV text here.
        """

        if isinstance(representation, str):
            representation = get_representation(representation, self.config.parser_tool, self.config.dpi)  # type: ignore[arg-type]
        page_set = conditioner.condition(question, self.page_count(question))
        if self.prediction_cache is not None:
            prediction_key = self._prediction_key(question, conditioner, representation.modality, page_set.page_indices)
            if self.prediction_cache.get(prediction_key) is not None:
                return page_set
        self.render_pages(question, page_set)
        return page_set

    def run_cell(
        self,
        question: Question,
        conditioner: InputConditioner,
        representation: Representation | str,
        prompt_mode: str = DEFAULT_PROMPT_MODE,
    ) -> PredictionRow:
        """Run (or fetch from cache) one `(question, condition, representation)` cell.

        Returns the cell's unjudged `PredictionRow`; scoring happens in a separate
        judge phase. `prompt_mode` selects the reasoner's instruction preamble; the
        cell's condition name already carries the mode, so the mode is part of the
        cache key. Setting it here (per cell) is what makes the same page set
        produce a distinct cell under each prompt in the hallucination sweep.
        """

        if isinstance(representation, str):
            representation = get_representation(representation, self.config.parser_tool, self.config.dpi)  # type: ignore[arg-type]
        if prompt_mode not in PROMPT_MODES:
            raise ValueError(f"unknown prompt_mode {prompt_mode!r}; expected one of {sorted(PROMPT_MODES)}")
        self.reasoner.prompt_instruction = PROMPT_MODES[prompt_mode]

        page_set = conditioner.condition(question, self.page_count(question))
        prediction_key = self._prediction_key(
            question, conditioner, representation.modality, page_set.page_indices
        )

        if self.prediction_cache is not None:
            cached = self.prediction_cache.get(prediction_key)
            if cached is not None:
                return cached

        pages = self.render_pages(question, page_set)
        payload = representation.build(pages)
        model_input = ModelInput.from_payload(payload)
        prediction: Prediction = self.reasoner.answer(question, model_input)

        row = self._prediction_row(question, conditioner, representation, page_set, prediction, prediction_key)
        if self.prediction_cache is not None:
            self.prediction_cache.put(row)
        return row

    def _prediction_row(
        self,
        question: Question,
        conditioner: InputConditioner,
        representation: Representation,
        page_set: PageSet,
        prediction: Prediction,
        prediction_key: str,
    ) -> PredictionRow:
        """Build the ok `PredictionRow` for a cell from its reasoner output."""

        return PredictionRow(
            prediction_key=prediction_key,
            question_id=question.id,
            doc_id=question.doc_id,
            doc_type=question.doc_type,
            bin_label=question.bin_label,
            scan_label=question.scan_label,
            hop=question.hop,
            is_unanswerable=question.is_unanswerable,
            evidence_sources=question.evidence_sources,
            condition=conditioner.name,
            provenance=page_set.provenance,
            page_indices=page_set.page_indices,
            representation=representation.modality,
            model_spec=prediction.model_spec or self.reasoner.spec,
            machine=self.machine,
            status="ok",
            skipped_reason="",
            oom_occurred=False,
            answer=prediction.text,
            total_text_tokens=prediction.total_text_tokens,
            total_visual_tokens=prediction.total_visual_tokens,
            text_tokens_fed=prediction.text_tokens_fed,
            output_tokens=prediction.output_tokens,
            tokens_dropped=tokens_dropped(prediction.total_text_tokens, prediction.text_tokens_fed),
            truncation_occurred=truncation_occurred(prediction.total_text_tokens, prediction.text_tokens_fed),
            latency_s=prediction.latency_s,
            prefill_latency_s=prediction.prefill_latency_s,
            decode_latency_s=prediction.decode_latency_s,
            peak_vram_bytes=prediction.peak_vram_bytes,
            visual_resolution=self.visual_resolution,
            note=page_set.note,
            metadata={"source_dataset": question.raw_fields.get("source_dataset", self.config.dataset)},
        )
