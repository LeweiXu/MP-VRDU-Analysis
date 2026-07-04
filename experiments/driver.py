"""Run experiments in two phases: generate on a GPU, judge/build anywhere.

Purpose:
    The execution engine behind the per-experiment modules. It splits every run
    so the GPU-only work (reasoner, retrievers, classifier) and the internet-only
    work (the judge) never have to happen on the same machine:

    - `generate(exp, config)` — GPU, offline-safe. Runs each experiment's cells
      (once per reasoner spec) and side work, caching predictions per experiment.
    - `judge(exp, config)` — no GPU, no PDFs. Re-scores the cached predictions
      with a real judge and writes judged rows.
    - `build(exp, config)` — pure aggregation. Loads the experiment's rows (plus
      its `depends_on` rows) and writes its table CSV(s).

    On Kaya you `submit` generate, `pull`, then judge+build locally. Locally you
    run all three in one process. Each experiment caches under its own directory
    (`results/cache/<smoke|full>/<name>/`) so one table re-runs in isolation.

Pipeline role:
    Sits between `registry.py` and the CLIs (`cli/experiments.py`,
    `kaya/generate.py`). It owns the per-experiment cache layout, the phase-2
    retriever/reasoner guards, and nothing about individual table shapes.

Arguments:
    None. Import-only; callers use `run_generate`, `run_judge`, `run_all`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from config import ExperimentConfig
from covariates.retriever import (
    BM25BGERetriever,
    ColQwenRetriever,
    MemoizedRetriever,
    Retriever,
)
from experiments.base import Experiment, Retrievers
from experiments.registry import resolve
from experiments.tables import TABLE_FILENAMES, load_result_rows
from pipeline.judge import Judge, StubJudge, get_judge
from pipeline.orchestrator import Orchestrator, PredictionCache, ResultCache, ResultRow
from pipeline.reasoner import Reasoner
from schema import Prediction, Question


def mode(config: ExperimentConfig) -> str:
    """Return the cache-partition name for this config."""

    return "smoke" if config.smoke else "full"


@dataclass(frozen=True)
class ExperimentPaths:
    """Per-experiment cache/side/table locations, all root-relative."""

    root: Path
    predictions: Path
    generate_results: Path
    results: Path
    side_dir: Path
    table_dir: Path


def experiment_paths(config: ExperimentConfig, name: str) -> ExperimentPaths:
    """Resolve the cache/table paths for one experiment."""

    root = config.paths.cache_dir / mode(config) / name
    return ExperimentPaths(
        root=root,
        predictions=root / "predictions.jsonl",
        generate_results=root / "generate_results.jsonl",
        results=root / "results.jsonl",
        side_dir=root,
        table_dir=config.paths.results_dir / "tables" / mode(config),
    )


class _GuardRetriever(Retriever):
    """Judge-phase retriever that must never run (cells must be cache hits)."""

    def __init__(self, name: str) -> None:
        self.name = name

    def retrieve(self, question: Question, page_count: int, k: int) -> tuple[int, ...]:
        raise RuntimeError(
            f"retriever {self.name!r} was called in the judge phase for "
            f"question {question.id!r}: run the generate phase first so this "
            "cell is a prediction-cache hit"
        )


class _SpecOnlyReasoner(Reasoner):
    """Judge-phase reasoner that only carries a spec; answering is a bug."""

    def __init__(self, spec: str) -> None:
        self.spec = spec

    def answer(self, question: Question, model_input) -> Prediction:  # noqa: ANN001
        raise RuntimeError(
            f"reasoner {self.spec!r} was called in the judge phase for "
            f"question {question.id!r}: the prediction must be cached from generate"
        )


def real_retrievers(config: ExperimentConfig) -> Retrievers:
    """Build the real retrievers (lazy: weights load only when first used)."""

    return Retrievers(
        text=MemoizedRetriever(
            BM25BGERetriever(data_dir=config.paths.data_dir, cache_dir=config.paths.cache_dir, dpi=config.dpi)
        ),
        vision=MemoizedRetriever(
            ColQwenRetriever(data_dir=config.paths.data_dir, cache_dir=config.paths.cache_dir, dpi=config.dpi)
        ),
    )


def guard_retrievers() -> Retrievers:
    """Retrievers that raise if used — for the judge phase."""

    return Retrievers(text=_GuardRetriever("bm25_bge_text"), vision=_GuardRetriever("colqwen_vision"))


def _reasoner_for(spec: str):
    """Return a real reasoner backend for the generate phase."""

    from models import get_reasoner

    return get_reasoner(spec)


def generate(config: ExperimentConfig, exp: Experiment, questions: Sequence[Question]) -> None:
    """Phase 1 (GPU): cache every prediction for one experiment, run side work."""

    paths = experiment_paths(config, exp.name)
    prediction_cache = PredictionCache(paths.predictions)
    generate_cache = ResultCache(paths.generate_results)
    retrievers = real_retrievers(config)
    exp_questions = exp.resolve_questions(config, questions)

    for spec in exp.model_specs(config):
        orchestrator = Orchestrator(
            config,
            reasoner=_reasoner_for(spec),
            judge=StubJudge("generate-throwaway"),
            cache=generate_cache,
            prediction_cache=prediction_cache,
        )
        for cell in exp.generation_cells(config, exp_questions, retrievers=retrievers):
            orchestrator.run_cell(cell.question, cell.conditioner, cell.representation)

    exp.run_side(config, exp_questions, paths.side_dir)


def judge(
    config: ExperimentConfig,
    exp: Experiment,
    questions: Sequence[Question],
    *,
    judge_impl: Judge,
) -> None:
    """Phase 2 (no GPU): re-judge the cached predictions for one experiment."""

    paths = experiment_paths(config, exp.name)
    prediction_cache = PredictionCache(paths.predictions)
    result_cache = ResultCache(paths.results)
    guards = guard_retrievers()
    exp_questions = exp.resolve_questions(config, questions)

    specs = exp.model_specs(config)
    if specs and len(prediction_cache) == 0:
        raise SystemExit(
            f"{exp.name}: no cached predictions at {paths.predictions}; run the "
            "generate phase first (kaya/generate.py or cli.experiments --phase generate)"
        )

    for spec in specs:
        orchestrator = Orchestrator(
            config,
            reasoner=_SpecOnlyReasoner(spec),
            judge=judge_impl,
            cache=result_cache,
            prediction_cache=prediction_cache,
        )
        for cell in exp.generation_cells(config, exp_questions, retrievers=guards):
            orchestrator.run_cell(cell.question, cell.conditioner, cell.representation)


def build(config: ExperimentConfig, exp: Experiment) -> dict[str, Path]:
    """Build one experiment's table CSV(s) from its rows + its dependencies' rows."""

    paths = experiment_paths(config, exp.name)
    rows: list[ResultRow] = list(load_result_rows(paths.results))
    for dep in exp.depends_on:
        rows.extend(load_result_rows(experiment_paths(config, dep).results))

    tables = exp.build(config, rows, paths.side_dir)
    paths.table_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for key, frame in tables.items():
        out = paths.table_dir / TABLE_FILENAMES[key]
        frame.to_csv(out, index=False)
        written[key] = out
    return written


def run_generate(config: ExperimentConfig, selector: str, questions: Sequence[Question]) -> None:
    """Generate one experiment or a group (in dependency order)."""

    for exp in resolve(selector):
        generate(config, exp, questions)


def run_judge(
    config: ExperimentConfig,
    selector: str,
    questions: Sequence[Question],
    *,
    judge_impl: Judge | None = None,
) -> dict[str, Path]:
    """Judge + build one experiment or a group; returns all written table paths."""

    judge_impl = judge_impl or get_judge("gemini")
    written: dict[str, Path] = {}
    for exp in resolve(selector):
        judge(config, exp, questions, judge_impl=judge_impl)
        written.update(build(config, exp))
    return written


def run_all(
    config: ExperimentConfig,
    selector: str,
    questions: Sequence[Question],
    *,
    judge_impl: Judge | None = None,
) -> dict[str, Path]:
    """Generate then judge+build in one process (a machine with GPU + internet)."""

    run_generate(config, selector, questions)
    return run_judge(config, selector, questions, judge_impl=judge_impl)
