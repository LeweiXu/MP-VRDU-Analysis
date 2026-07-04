"""Run the whole paper end to end on the smoke corpus and emit all 8 tables.

Purpose:
    This is the Section-1 MVP "full run, tiny data" driver. It exercises the
    real pipeline (real Qwen3-VL-2B reasoner, real BM25+BGE / ColQwen
    retrievers, real Qwen doc-type classifier, real GPT-4o-mini judge) over the
    frozen ~7-document smoke corpus and produces the eight paper table CSVs
    filled with real-but-throwaway numbers. No stub reasoners, no injected
    scorers: the point is to prove the real components run together, cheaply.

Two phases (identical locally and on Kaya):
    The reasoner/retrievers/classifier need a GPU; the GPT-4o-mini judge needs
    the internet. On Kaya those never coexist (compute node = GPU, no internet;
    login node = internet, no GPU), so the run is split into two phases that
    share a `PredictionCache`:

    - `generate` (GPU, offline-safe): run every cell's A->C path, cache the
      reasoner predictions; run the classifier once per document and log it.
    - `judge` (internet, no GPU): re-judge the cached predictions with
      GPT-4o-mini (the reasoner never runs again) and build the 8 tables.

    Locally you just run both phases in one env (`--phase all`); on Kaya
    `generate` is a GPU `submit` job and `judge` is a login-node `run`.

Pipeline role:
    Composes the frozen orchestrator, the M4 oracle ladder, and the M6
    matched/cross + routing covariates into one reproducible smoke sweep, then
    calls `experiments.tables` to write the CSVs. It changes no frozen
    interface; it only wires the existing pieces together with the two-layer
    cache.

Arguments:
    None. Import-only module. Callers use `run_generate(config)`,
    `run_judge(config)`, or `run_all(config)`. The `cli.run_smoke` and
    `kaya/smoke_generate.py` / `kaya/smoke_judge.py` entry points drive it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from config import ExperimentConfig
from covariates.classifier import DocTypePrediction, QwenDocTypeClassifier
from covariates.retriever import (
    BM25BGERetriever,
    ColQwenRetriever,
    MemoizedRetriever,
    Retriever,
)
from data.binning import doc_type_bin
from experiments.smoke import load_smoke_questions
from experiments.tables import write_all_tables
from metrics.retrieval import RetrievalEvalRow, score_retrieval
from pipeline.conditioner import InputConditioner, OracleConditioner, RetrievedTopK
from pipeline.judge import GPT4oMiniJudge, Judge, StubJudge
from pipeline.orchestrator import (
    Orchestrator,
    PredictionCache,
    ResultCache,
    ResultRow,
)
from schema import Modality, Question


# The smoke run's cache/table layout, all root-relative under results/.
SMOKE_SUBDIR = "smoke"
MATCHED_CROSS_REPRESENTATION: Modality = "TLV"


@dataclass(frozen=True)
class SmokePaths:
    """Resolved cache/log/table paths for a smoke run."""

    prediction_cache: Path
    generate_cache: Path
    result_cache: Path
    classifier_log: Path
    retrieval_log: Path
    table_dir: Path


def smoke_paths(config: ExperimentConfig) -> SmokePaths:
    """Return the root-relative smoke artifact paths for a config."""

    cache_root = config.paths.cache_dir / SMOKE_SUBDIR
    return SmokePaths(
        prediction_cache=cache_root / "predictions.jsonl",
        generate_cache=cache_root / "generate_results.jsonl",
        result_cache=cache_root / "results.jsonl",
        classifier_log=cache_root / "classifier.jsonl",
        retrieval_log=cache_root / "retrieval.jsonl",
        table_dir=config.paths.results_dir / "tables" / SMOKE_SUBDIR,
    )


class _CachedOnlyRetriever(Retriever):
    """Judge-phase guard: retrieval must already be in the prediction cache.

    The judge phase runs on a machine with no GPU. Every retrieved cell must
    therefore be a prediction-cache hit (so the conditioner is never asked to
    retrieve). If it is ever called, that means the generate phase did not cover
    this cell, and we want a clear error rather than a silent CPU/GPU attempt.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def retrieve(self, question: Question, page_count: int, k: int) -> tuple[int, ...]:
        raise RuntimeError(
            f"retriever {self.name!r} was called in the judge phase for "
            f"question {question.id!r}: the generate phase must run first so "
            "this cell is a prediction-cache hit"
        )


def _matched_cross_conditioners(
    config: ExperimentConfig,
    *,
    text_retriever: Retriever,
    vision_retriever: Retriever,
    k: int,
) -> tuple[tuple[str, str, Retriever, InputConditioner], ...]:
    """Return the (pipeline, modality, retriever, conditioner) tuples for T6.

    Matched = vision retrieval + vision reasoning; cross = text retrieval +
    vision reasoning. Conditioner names are fixed strings so the prediction key
    is identical across the generate and judge phases.
    """

    return (
        (
            "matched_vision",
            "vision",
            vision_retriever,
            RetrievedTopK(vision_retriever, k, name=f"retrieved_vision_k{k}"),
        ),
        (
            "cross_text_to_vision",
            "text",
            text_retriever,
            RetrievedTopK(text_retriever, k, name=f"retrieved_text_k{k}"),
        ),
    )


def _oracle_cells(
    config: ExperimentConfig, questions: Sequence[Question]
) -> list[tuple[Question, InputConditioner, Modality]]:
    """Oracle ladder cells (T/TL/TLV/V) that feed Tables 1-5, 7, 8."""

    oracle = OracleConditioner()
    cells: list[tuple[Question, InputConditioner, Modality]] = []
    for question in questions:
        for rung in config.representations:
            cells.append((question, oracle, rung))
    return cells


def _load_questions(config: ExperimentConfig, questions: Iterable[Question] | None) -> list[Question]:
    """Return the smoke questions to run (frozen corpus by default)."""

    if questions is not None:
        return list(questions)
    return list(load_smoke_questions(config.paths.data_dir))


def _emit(event: dict[str, object]) -> None:
    """Print one machine-readable JSON progress line."""

    print(json.dumps(event, sort_keys=True), flush=True)


def run_generate(
    config: ExperimentConfig,
    questions: Iterable[Question] | None = None,
    *,
    k: int | None = None,
) -> SmokePaths:
    """Phase 1: generate + cache every prediction on the GPU, log the classifier.

    Runs offline-safe: the only components used here are the local reasoner,
    the two retrievers, and the doc-type classifier. Results are judged with a
    throwaway `StubJudge` purely so the orchestrator can complete a row; the
    durable artifact is the prediction cache the judge phase reads.
    """

    paths = smoke_paths(config)
    question_list = _load_questions(config, questions)
    top_k = int(k if k is not None else (config.k_values[0] if config.k_values else 1))

    prediction_cache = PredictionCache(paths.prediction_cache)
    orchestrator = Orchestrator(
        config,
        judge=StubJudge("generate-throwaway"),
        cache=ResultCache(paths.generate_cache),
        prediction_cache=prediction_cache,
    )

    text_retriever = MemoizedRetriever(
        BM25BGERetriever(
            data_dir=config.paths.data_dir,
            cache_dir=config.paths.cache_dir,
            dpi=config.dpi,
        )
    )
    vision_retriever = MemoizedRetriever(
        ColQwenRetriever(
            data_dir=config.paths.data_dir,
            cache_dir=config.paths.cache_dir,
            dpi=config.dpi,
        )
    )

    _emit(
        {
            "event": "generate_start",
            "questions": len(question_list),
            "representations": list(config.representations),
            "reasoner_spec": orchestrator.reasoner.spec,
            "prediction_cache": str(paths.prediction_cache),
        }
    )

    # Oracle ladder -> Tables 1-5, 7, 8.
    oracle_cells = _oracle_cells(config, question_list)
    for question, conditioner, rung in oracle_cells:
        orchestrator.run_cell(question, conditioner, rung)

    # Matched/cross retrieval -> Table 6, plus retrieval R/P/F1 metrics.
    retrieval_rows: list[RetrievalEvalRow] = []
    pipelines = _matched_cross_conditioners(
        config, text_retriever=text_retriever, vision_retriever=vision_retriever, k=top_k
    )
    for question in question_list:
        page_count = orchestrator.page_count(question)
        for _pipeline, modality, retriever, conditioner in pipelines:
            ranked = retriever.retrieve(question, page_count, top_k)
            retrieval_rows.append(
                score_retrieval(question, ranked, retriever=retriever.name, modality=modality, k=top_k)
            )
            orchestrator.run_cell(question, conditioner, MATCHED_CROSS_REPRESENTATION)
    _write_retrieval_log(paths.retrieval_log, retrieval_rows)

    # Doc-type classifier -> routing latency/accuracy (RQ3 covariate), one per doc.
    classifier = QwenDocTypeClassifier(
        data_dir=config.paths.data_dir,
        cache_dir=config.paths.cache_dir,
        dpi=config.dpi,
    )
    classifier_records = _run_classifier(classifier, question_list)
    _write_classifier_log(paths.classifier_log, classifier_records)

    _emit(
        {
            "event": "generate_complete",
            "predictions_cached": len(prediction_cache),
            "oracle_cells": len(oracle_cells),
            "matched_cross_cells": len(question_list) * len(pipelines),
            "classified_docs": len(classifier_records),
            "prediction_cache": str(paths.prediction_cache),
            "classifier_log": str(paths.classifier_log),
        }
    )
    return paths


def run_judge(
    config: ExperimentConfig,
    questions: Iterable[Question] | None = None,
    *,
    judge: Judge | None = None,
    k: int | None = None,
    n_bootstrap: int = 200,
) -> SmokePaths:
    """Phase 2: judge cached predictions with GPT-4o-mini and write 8 tables.

    Uses no GPU and opens no PDFs: every cell must be a prediction-cache hit
    from `run_generate`. The judge is the real GPT-4o-mini API judge by default.
    """

    paths = smoke_paths(config)
    question_list = _load_questions(config, questions)
    top_k = int(k if k is not None else (config.k_values[0] if config.k_values else 1))

    prediction_cache = PredictionCache(paths.prediction_cache)
    if len(prediction_cache) == 0:
        raise SystemExit(
            f"no cached predictions at {paths.prediction_cache}; run the generate "
            "phase first (kaya/smoke_generate.py or cli.run_smoke --phase generate)"
        )

    real_judge = judge or GPT4oMiniJudge()
    orchestrator = Orchestrator(
        config,
        judge=real_judge,
        cache=ResultCache(paths.result_cache),
        prediction_cache=prediction_cache,
    )

    text_guard = _CachedOnlyRetriever("bm25_bge_text")
    vision_guard = _CachedOnlyRetriever("colqwen_vision")
    pipelines = _matched_cross_conditioners(
        config, text_retriever=text_guard, vision_retriever=vision_guard, k=top_k
    )

    _emit(
        {
            "event": "judge_start",
            "questions": len(question_list),
            "judge_spec": real_judge.spec,
            "predictions_cached": len(prediction_cache),
        }
    )

    rows: list[ResultRow] = []
    for question, conditioner, rung in _oracle_cells(config, question_list):
        rows.append(orchestrator.run_cell(question, conditioner, rung))
    for question in question_list:
        for _pipeline, _modality, _retriever, conditioner in pipelines:
            rows.append(orchestrator.run_cell(question, conditioner, MATCHED_CROSS_REPRESENTATION))

    table_paths = write_all_tables(
        rows,
        paths.table_dir,
        dataset=config.dataset,
        bins=config.bins,
        margin_points=config.sufficiency_margin,
        n_bootstrap=n_bootstrap,
    )

    _emit(
        {
            "event": "judge_complete",
            "rows": len(rows),
            "correct": sum(1 for row in rows if row.correct),
            "result_cache": str(paths.result_cache),
            "tables": {name: str(path) for name, path in table_paths.items()},
        }
    )
    return paths


def run_all(
    config: ExperimentConfig,
    questions: Iterable[Question] | None = None,
    *,
    judge: Judge | None = None,
    k: int | None = None,
) -> SmokePaths:
    """Run both phases in one process (local GPU + internet)."""

    run_generate(config, questions, k=k)
    return run_judge(config, questions, judge=judge, k=k)


def _run_classifier(
    classifier: QwenDocTypeClassifier, questions: Sequence[Question]
) -> list[dict[str, object]]:
    """Classify each distinct document once and return log records."""

    seen: set[str] = set()
    records: list[dict[str, object]] = []
    for question in questions:
        if question.doc_id in seen:
            continue
        seen.add(question.doc_id)
        prediction = classifier.classify(question)
        records.append(_classifier_record(question, prediction, classifier.name))
    return records


def _classifier_record(
    question: Question, prediction: DocTypePrediction, classifier_name: str
) -> dict[str, object]:
    """Serialise one classifier prediction with gold/predicted bins."""

    gold_bin = doc_type_bin(question.doc_type)
    predicted_bin = str(prediction.bin or gold_bin)
    return {
        "doc_id": question.doc_id,
        "gold_doc_type": question.doc_type,
        "predicted_doc_type": prediction.doc_type,
        "gold_bin": gold_bin,
        "predicted_bin": predicted_bin,
        "correct_bin": predicted_bin == gold_bin,
        "confidence": prediction.confidence,
        "latency_s": prediction.latency_s,
        "classifier": classifier_name,
        "raw_text": prediction.raw_text,
    }


def _write_classifier_log(path: Path, records: Sequence[dict[str, object]]) -> None:
    """Write classifier predictions as jsonl."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _write_retrieval_log(path: Path, rows: Sequence[RetrievalEvalRow]) -> None:
    """Write per-question retrieval R/P/F1 rows as jsonl."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(_retrieval_record(row), sort_keys=True) + "\n")


def _retrieval_record(row: RetrievalEvalRow) -> dict[str, object]:
    """Serialise one retrieval evaluation row to a plain dict."""

    record = asdict(row)
    for key, value in list(record.items()):
        if isinstance(value, tuple):
            record[key] = list(value)
    return record
