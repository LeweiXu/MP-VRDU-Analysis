"""Composes the five stages of one cell and owns the two cache layers and telemetry capture."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from config import DEFAULT_PROMPT_MODE, PROMPT_MODES, ExperimentConfig
from data.loader import resolve_pdf
from data.render import pdf_page_count, render_pdf
from experiments.engine.paths import prediction_key as make_prediction_key
from experiments.engine.paths import result_key as make_result_key
from models import get_reasoner
from models.payload import ModelInput
from pipeline.conditioner import InputConditioner
from pipeline.judge import Judge, StubJudge, get_judge
from pipeline.reasoner import Reasoner
from pipeline.representation import Representation, get_representation
from schema import Page, PageSet, Prediction, Question, ResultRow, Score, tokens_dropped, truncation_occurred

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


@dataclass(frozen=True)
class CachedPrediction:
    """A reasoner output plus its page provenance, keyed without the judge.

    This is the durable record produced by the reasoner stage. It carries
    everything a `ResultRow` needs from the reasoner path so a later judge-only
    pass can rebuild the row without re-running the model.
    """

    prediction_key: str
    question_id: str
    doc_id: str
    condition: str
    representation: str
    model_spec: str
    provenance: str
    page_indices: tuple[int, ...]
    note: str
    text: str
    total_text_tokens: int
    total_visual_tokens: int
    text_tokens_fed: int
    output_tokens: int
    latency_s: float
    prefill_latency_s: float
    decode_latency_s: float
    peak_vram_bytes: int
    visual_resolution: str = ""

    def to_json(self) -> str:
        data = asdict(self)
        data["page_indices"] = list(self.page_indices)
        return json.dumps(data, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CachedPrediction":
        data = dict(data)
        data["page_indices"] = tuple(data.get("page_indices", ()))
        return cls(**data)

    def as_prediction(self) -> Prediction:
        """Rebuild the frozen `Prediction` this record was serialised from."""

        return Prediction(
            text=self.text,
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


class PredictionCache:
    """Append-only jsonl cache of reasoner outputs keyed without the judge."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._index: dict[str, CachedPrediction] = {}
        self._handle = None
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                record = CachedPrediction.from_dict(json.loads(line))
                self._index[record.prediction_key] = record

    def get(self, prediction_key: str) -> CachedPrediction | None:
        return self._index.get(prediction_key)

    def put(self, record: CachedPrediction) -> None:
        if record.prediction_key in self._index:
            return
        self._index[record.prediction_key] = record
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a")
        self._handle.write(record.to_json() + "\n")
        self._handle.flush()

    def __iter__(self) -> Iterator[CachedPrediction]:
        return iter(self._index.values())

    def __len__(self) -> int:
        return len(self._index)


class Orchestrator:
    """Compose the pipeline for one cell, with the two cache layers."""

    def __init__(
        self,
        config: ExperimentConfig,
        reasoner: Reasoner | None = None,
        judge: Judge | None = None,
        cache: ResultCache | None = None,
        prediction_cache: PredictionCache | None = None,
        machine: str | None = None,
        visual_resolution: str | None = None,
    ) -> None:
        self.config = config
        self.reasoner = reasoner if reasoner is not None else get_reasoner(config.reasoner_spec)
        self.judge = judge if judge is not None else get_judge(config.judge_spec)
        cache_path = config.paths.cache_dir / "orchestrator" / "results.jsonl"
        self.cache = cache if cache is not None else ResultCache(cache_path)
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

    def _keys(self, question: Question, conditioner: InputConditioner, modality: str,
              page_indices: tuple[int, ...]) -> tuple[str, str]:
        prediction_key = make_prediction_key(
            question.id, question.doc_id, conditioner.name, modality, self.reasoner.spec,
            page_indices, self.visual_resolution,
        )
        result_key = make_result_key(
            question.id, question.doc_id, conditioner.name, modality, self.reasoner.spec,
            page_indices, self.judge.spec, self.visual_resolution,
        )
        return prediction_key, result_key

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
            prediction_key, _ = self._keys(question, conditioner, representation.modality, page_set.page_indices)
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
    ) -> ResultRow:
        """Run (or fetch from cache) one `(question, condition, representation)` cell.

        `prompt_mode` selects the reasoner's instruction preamble; the cell's
        condition name already carries the mode, so the mode is part of the cache
        key. Setting it here (per cell) is what makes the same page set produce a
        distinct cell under each prompt in the hallucination sweep.
        """

        if isinstance(representation, str):
            representation = get_representation(representation, self.config.parser_tool, self.config.dpi)  # type: ignore[arg-type]
        if prompt_mode not in PROMPT_MODES:
            raise ValueError(f"unknown prompt_mode {prompt_mode!r}; expected one of {sorted(PROMPT_MODES)}")
        self.reasoner.prompt_instruction = PROMPT_MODES[prompt_mode]

        page_set = conditioner.condition(question, self.page_count(question))
        prediction_key, result_key = self._keys(
            question, conditioner, representation.modality, page_set.page_indices
        )

        cached = self.cache.get(result_key)
        if cached is not None:
            return cached

        prediction = self._resolve_prediction(question, conditioner, representation, page_set, prediction_key)
        score: Score = self.judge.score(question, prediction)

        dropped = tokens_dropped(prediction.total_text_tokens, prediction.text_tokens_fed)
        row = ResultRow(
            result_key=result_key,
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
            judge_spec=score.judge_spec or self.judge.spec,
            machine=self.machine,
            status="ok",
            skipped_reason="",
            oom_occurred=False,
            answer=prediction.text,
            score=score.value,
            correct=score.correct,
            abstained=score.abstained,
            total_text_tokens=prediction.total_text_tokens,
            total_visual_tokens=prediction.total_visual_tokens,
            text_tokens_fed=prediction.text_tokens_fed,
            output_tokens=prediction.output_tokens,
            tokens_dropped=dropped,
            truncation_occurred=truncation_occurred(prediction.total_text_tokens, prediction.text_tokens_fed),
            latency_s=prediction.latency_s,
            prefill_latency_s=prediction.prefill_latency_s,
            decode_latency_s=prediction.decode_latency_s,
            peak_vram_bytes=prediction.peak_vram_bytes,
            visual_resolution=self.visual_resolution,
            note=page_set.note,
            metadata={"source_dataset": question.raw_fields.get("source_dataset", self.config.dataset)},
        )
        self.cache.put(row)
        return row

    def _resolve_prediction(
        self,
        question: Question,
        conditioner: InputConditioner,
        representation: Representation,
        page_set: PageSet,
        prediction_key: str,
    ) -> Prediction:
        """Return a prediction, using the prediction cache to skip the model."""

        if self.prediction_cache is not None:
            hit = self.prediction_cache.get(prediction_key)
            if hit is not None:
                return hit.as_prediction()

        pages = self.render_pages(question, page_set)
        payload = representation.build(pages)
        model_input = ModelInput.from_payload(payload)
        prediction: Prediction = self.reasoner.answer(question, model_input)

        if self.prediction_cache is not None:
            self.prediction_cache.put(
                CachedPrediction(
                    prediction_key=prediction_key,
                    question_id=question.id,
                    doc_id=question.doc_id,
                    condition=conditioner.name,
                    representation=representation.modality,
                    model_spec=prediction.model_spec or self.reasoner.spec,
                    provenance=page_set.provenance,
                    page_indices=page_set.page_indices,
                    note=page_set.note,
                    text=prediction.text,
                    total_text_tokens=prediction.total_text_tokens,
                    total_visual_tokens=prediction.total_visual_tokens,
                    text_tokens_fed=prediction.text_tokens_fed,
                    output_tokens=prediction.output_tokens,
                    latency_s=prediction.latency_s,
                    prefill_latency_s=prediction.prefill_latency_s,
                    decode_latency_s=prediction.decode_latency_s,
                    peak_vram_bytes=prediction.peak_vram_bytes,
                    visual_resolution=self.visual_resolution,
                )
            )
        return prediction
