"""End-to-end pipeline orchestration and result caching entry point.

The orchestrator composes the four pipeline stages for one
`(question, condition, representation)` cell and returns a well-typed
`ResultRow`:

    conditioner (A) -> render pages -> representation (B) -> ModelInput
        -> reasoner (C) -> judge (D)

It also owns the **caching contract** frozen at Stage 3: every cell is keyed by a
deterministic hash of its inputs (question, doc, condition, representation,
reasoner spec, judge spec, dpi) and written to a jsonl cache under
`results/cache/`. Re-running is idempotent and resumable, which is the only way
the multi-condition sweep is affordable. Nothing in this file knows which real
tools or models sit behind the ABCs; swapping them never changes the run loop.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from config import DEFAULT_PATHS, ExperimentConfig
from data.loader import resolve_pdf
from data.render import pdf_page_count, render_pdf
from models import get_reasoner
from models.payload import ModelInput
from pipeline.conditioner import InputConditioner
from pipeline.judge import Judge, StubJudge
from pipeline.reasoner import Reasoner
from pipeline.representation import Representation, get_representation
from schema import Page, PageSet, Prediction, Question, Score


@dataclass(frozen=True)
class ResultRow:
    """One cached, well-typed pipeline result for a single cell."""

    cache_key: str
    question_id: str
    doc_id: str
    doc_type: str
    hop: str
    is_unanswerable: bool
    evidence_sources: tuple[str, ...]
    condition: str
    provenance: str
    page_indices: tuple[int, ...]
    representation: str
    model_spec: str
    judge_spec: str
    answer: str
    input_text_tokens: int
    input_visual_tokens: int
    output_tokens: int
    latency_s: float
    score: float
    correct: bool
    abstained: bool
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


class ResultCache:
    """Append-only jsonl cache keyed by cache_key, resumable across runs."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._index: dict[str, ResultRow] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                row = ResultRow.from_dict(json.loads(line))
                self._index[row.cache_key] = row

    def get(self, cache_key: str) -> ResultRow | None:
        return self._index.get(cache_key)

    def put(self, row: ResultRow) -> None:
        if row.cache_key in self._index:
            return
        self._index[row.cache_key] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(row.to_json() + "\n")

    def __iter__(self) -> Iterator[ResultRow]:
        return iter(self._index.values())

    def __len__(self) -> int:
        return len(self._index)


def make_cache_key(
    question: Question,
    condition_name: str,
    representation: str,
    model_spec: str,
    judge_spec: str,
    dpi: int,
) -> str:
    """Deterministic hash of everything a cell's result depends on."""

    payload = json.dumps(
        {
            "question_id": question.id,
            "doc_id": question.doc_id,
            "condition": condition_name,
            "representation": representation,
            "model_spec": model_spec,
            "judge_spec": judge_spec,
            "dpi": dpi,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class Orchestrator:
    """Compose the pipeline for one cell, with result caching."""

    def __init__(
        self,
        config: ExperimentConfig,
        reasoner: Reasoner | None = None,
        judge: Judge | None = None,
        cache: ResultCache | None = None,
    ) -> None:
        self.config = config
        self.reasoner = reasoner or get_reasoner(config.reasoner_spec)
        self.judge = judge or StubJudge(config.judge_spec)
        cache_path = config.paths.cache_dir / "orchestrator" / "results.jsonl"
        self.cache = cache or ResultCache(cache_path)
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
        return render_pdf(
            pdf,
            page_set.page_indices,
            cache_dir=self.config.paths.cache_dir,
            dpi=self.config.dpi,
        )

    # -- the run loop -----------------------------------------------------

    def run_cell(
        self,
        question: Question,
        conditioner: InputConditioner,
        representation: Representation | str,
    ) -> ResultRow:
        """Run (or fetch from cache) one `(question, condition, representation)` cell."""

        if isinstance(representation, str):
            representation = get_representation(representation)  # type: ignore[arg-type]

        cache_key = make_cache_key(
            question,
            conditioner.name,
            representation.modality,
            self.reasoner.spec,
            self.judge.spec,
            self.config.dpi,
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        page_set = conditioner.condition(question, self.page_count(question))
        pages = self.render_pages(question, page_set)
        payload = representation.build(pages)
        model_input = ModelInput.from_payload(payload)
        prediction: Prediction = self.reasoner.answer(question, model_input)
        score: Score = self.judge.score(question, prediction)

        row = ResultRow(
            cache_key=cache_key,
            question_id=question.id,
            doc_id=question.doc_id,
            doc_type=question.doc_type,
            hop=question.hop,
            is_unanswerable=question.is_unanswerable,
            evidence_sources=question.evidence_sources,
            condition=conditioner.name,
            provenance=page_set.provenance,
            page_indices=page_set.page_indices,
            representation=representation.modality,
            model_spec=prediction.model_spec or self.reasoner.spec,
            judge_spec=score.judge_spec or self.judge.spec,
            answer=prediction.text,
            input_text_tokens=prediction.input_text_tokens,
            input_visual_tokens=prediction.input_visual_tokens,
            output_tokens=prediction.output_tokens,
            latency_s=prediction.latency_s,
            score=score.value,
            correct=score.correct,
            abstained=score.abstained,
            metadata={"note": page_set.note},
        )
        self.cache.put(row)
        return row
