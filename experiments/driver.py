"""The task-execution engine: generate (GPU) and judge (local), over tasks.

Purpose:
    The runtime shared by the `cli/generate.py` and `cli/judge.py` wrappers. It
    runs a `GenerationTask`'s cells once per reasoner spec:

    - `generate` (GPU, offline): caches predictions per task + runs side work.
    - `judge` (no GPU, no PDFs): re-scores the cached predictions with a real
      judge; the guards raise `CacheMiss` if a cell was never generated.

    Table building is a separate role (`experiments/tables.py`); this engine only
    produces `predictions.jsonl` and `results.jsonl` per task.

Pipeline role:
    Sits between the registry (`experiments/registry.py`) + task files
    (`experiments/G*_*.py`) and the CLI wrappers. Owns the reasoner/retriever
    construction, the phase-2 guards, and the resumable per-task run loop.

Arguments:
    None. Import-only; callers use `run_generate`, `run_judge`, `config_from_args`.
"""

from __future__ import annotations

import argparse
import time
import traceback
from collections.abc import Sequence

from config import ExperimentConfig, max_input_tokens_for_spec, max_pixels_for_resolution
from covariates.retriever import BM25BGERetriever, ColQwenRetriever, MemoizedRetriever, Retriever
from experiments.base import GenerationTask, Retrievers
from experiments.paths import answer_preview, experiment_paths, free_gpu, log, mode, write_phase_status
from experiments.registry import resolve
from pipeline.judge import Judge, StubJudge, get_judge
from pipeline.orchestrator import Orchestrator, PredictionCache, ResultCache
from pipeline.reasoner import Reasoner
from schema import Prediction, Question


# ---------------------------------------------------------------------------
# GPU-side construction
# ---------------------------------------------------------------------------
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


def reasoner_for(spec: str, config: ExperimentConfig | None = None):
    """Return a real reasoner backend for the generate phase.

    Passes the config's generation cap and a vision-token cap so local VLM cells
    stay within GPU memory: an explicit `--visual-resolution` if set, else the
    size-aware default (a smaller `max_pixels` for the bigger reasoners). See
    `config.max_pixels_for_resolution`.
    """

    from models import get_reasoner

    if config is None:
        return get_reasoner(spec)
    max_pixels = max_pixels_for_resolution(spec, config)
    max_input_tokens = max_input_tokens_for_spec(spec, config.max_input_tokens)
    log.info(
        "building reasoner spec=%s max_new_tokens=%d max_pixels=%d (resolution=%s) max_input_tokens=%d",
        spec,
        config.max_tokens,
        max_pixels,
        config.visual_resolution or "size-aware",
        max_input_tokens,
    )
    return get_reasoner(
        spec,
        max_new_tokens=config.max_tokens,
        max_pixels=max_pixels,
        max_input_tokens=max_input_tokens,
    )


# ---------------------------------------------------------------------------
# Judge-phase guards
# ---------------------------------------------------------------------------
class CacheMiss(RuntimeError):
    """A judge-phase cell whose prediction/retrieval was never generated.

    The judge phase only re-scores cached predictions; it must never run a
    reasoner or retriever. When it reaches a cell that generate did not produce
    (e.g. a partial cache after an OOM), the guards below raise this. With
    `--continue-on-error` the judge skips such cells so a partial table still
    builds; without it, it surfaces as the usual hard error.
    """


class _GuardRetriever(Retriever):
    """Judge-phase retriever that must never run (cells must be cache hits)."""

    def __init__(self, name: str) -> None:
        self.name = name

    def retrieve(self, question: Question, page_count: int, k: int) -> tuple[int, ...]:
        raise CacheMiss(
            f"retriever {self.name!r} was called in the judge phase for "
            f"question {question.id!r}: run the generate phase first so this "
            "cell is a prediction-cache hit"
        )


class _SpecOnlyReasoner(Reasoner):
    """Judge-phase reasoner that only carries a spec; answering is a bug."""

    def __init__(self, spec: str) -> None:
        self.spec = spec

    def answer(self, question: Question, model_input) -> Prediction:  # noqa: ANN001
        raise CacheMiss(
            f"reasoner {self.spec!r} was called in the judge phase for "
            f"question {question.id!r}: the prediction must be cached from generate"
        )


def guard_retrievers() -> Retrievers:
    """Retrievers that raise if used — for the judge phase."""

    return Retrievers(text=_GuardRetriever("bm25_bge_text"), vision=_GuardRetriever("colqwen_vision"))


# ---------------------------------------------------------------------------
# Generate (GPU)
# ---------------------------------------------------------------------------
def generate(
    config: ExperimentConfig,
    task: GenerationTask,
    questions: Sequence[Question],
    *,
    skip_failed_cells: bool = False,
) -> None:
    """Phase 1 (GPU): cache every prediction for one task, then run side work.

    `skip_failed_cells` (wired to `--continue-on-error`) logs and skips a cell
    that raises (e.g. a rare many-gold-page cell whose O(seq^2) attention OOMs a
    V100 even after downscaling), freeing the GPU and continuing, instead of
    aborting the whole task. The skipped cells are simply absent from the cache;
    the judge (`--continue-on-error`) then scores what was generated.
    """

    paths = experiment_paths(config, task.name)
    prediction_cache = PredictionCache(paths.predictions)
    generate_cache = ResultCache(paths.generate_results)
    retrievers = real_retrievers(config)
    task_questions = task.resolve_questions(config, questions)
    specs = task.model_specs(config)

    log.info(
        "=== generate %s (mode=%s) | %d questions | specs=%s ===",
        task.name,
        mode(config),
        len(task_questions),
        list(specs) or "(side-only)",
    )

    reasoner = None
    total_cells = 0
    total_skipped = 0
    for spec in specs:
        reasoner = reasoner_for(spec, config)
        orchestrator = Orchestrator(
            config,
            reasoner=reasoner,
            judge=StubJudge("generate-throwaway"),
            cache=generate_cache,
            prediction_cache=prediction_cache,
        )
        cells = task.generation_cells(config, task_questions, retrievers=retrievers)
        log.info("%s spec=%s: %d cells to run", task.name, spec, len(cells))
        total_cells += len(cells)
        skipped_cells = 0

        # Parse pre-pass: warm the Marker/Surya (and retrieval) disk caches with
        # the reasoner NOT yet loaded, then free those model stacks. This is what
        # stops the parser and the reasoner from sharing VRAM on a 16GB V100.
        log.info("%s spec=%s: parse pre-pass (warming caches, reasoner not loaded)", task.name, spec)
        prewarm_started = time.perf_counter()
        for index, cell in enumerate(cells, start=1):
            try:
                orchestrator.prewarm_cell(cell.question, cell.conditioner, cell.representation)
            except Exception:
                log.error(
                    "prewarm FAILED %s cell %d/%d q=%s cond=%s rep=%s",
                    task.name, index, len(cells), cell.question.id, cell.conditioner.name, cell.representation,
                )
                raise
        retrievers.text.unload()
        retrievers.vision.unload()
        free_gpu()
        log.info("%s spec=%s: pre-pass done (%.1fs); GPU freed for reasoner", task.name, spec, time.perf_counter() - prewarm_started)

        for index, cell in enumerate(cells, start=1):
            label = (
                f"{task.name} spec={spec} cell {index}/{len(cells)} "
                f"q={cell.question.id} doc={cell.question.doc_id} "
                f"cond={cell.conditioner.name} rep={cell.representation}"
            )
            log.info("-> %s", label)
            started = time.perf_counter()
            try:
                row = orchestrator.run_cell(cell.question, cell.conditioner, cell.representation)
            except Exception as exc:
                log.error("FAILED %s (after %.1fs): %s: %s", label, time.perf_counter() - started, type(exc).__name__, exc)
                if not skip_failed_cells:
                    raise
                skipped_cells += 1
                free_gpu()  # recover activation memory (e.g. after a CUDA OOM) before the next cell
                continue
            log.info(
                "   done %.1fs | in_txt=%d in_vis=%d out=%d | correct=%s abstained=%s | ans=%r",
                time.perf_counter() - started,
                row.input_text_tokens,
                row.input_visual_tokens,
                row.output_tokens,
                row.correct,
                row.abstained,
                answer_preview(row.answer),
            )

        if skipped_cells:
            log.warning("%s spec=%s: skipped %d/%d cell(s) that failed (see FAILED lines above)", task.name, spec, skipped_cells, len(cells))
        total_skipped += skipped_cells

        # Release the reasoner before the next spec or the side work, so a
        # multi-spec task (or the classifier in run_side) starts from a clean GPU.
        if hasattr(reasoner, "free"):
            reasoner.free()
        del orchestrator
        reasoner = None
        free_gpu()

    # A task whose every reasoner cell failed produced no predictions, so its
    # "generate" phase must not report success (that false-success is how a
    # broken run, e.g. a missing `timm` for InternVL, slipped through as a table
    # dependency). Raise so run_generate records status=failed instead.
    if total_cells and total_skipped == total_cells:
        raise RuntimeError(
            f"{task.name}: all {total_cells} reasoner cell(s) failed and were skipped; "
            "no predictions were written (see FAILED lines above)"
        )

    log.info("%s: running side work in %s", task.name, paths.side_dir)
    started = time.perf_counter()
    task.run_side(config, task_questions, paths.side_dir)
    free_gpu()  # side work (retriever/classifier) also holds GPU weights
    log.info("%s: side work done (%.1fs)", task.name, time.perf_counter() - started)


def run_generate(
    config: ExperimentConfig,
    selector: str,
    questions: Sequence[Question],
    *,
    continue_on_error: bool = False,
) -> list:
    """Generate one task or a group; return per-task statuses."""

    statuses = []
    for task in resolve(selector):
        try:
            generate(config, task, questions, skip_failed_cells=continue_on_error)
        except Exception as exc:
            status = write_phase_status(config, task.name, phase="generate", status="failed", error=exc)
            statuses.append(status)
            log.error(
                "[generate] %s: FAILED (%s: %s)\n%s",
                task.name,
                status.error_type,
                status.error,
                "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            )
            if not continue_on_error:
                raise
            continue
        status = write_phase_status(config, task.name, phase="generate", status="success")
        statuses.append(status)
        log.info("[generate] %s: success -> %s", task.name, status.path)
    return statuses


# ---------------------------------------------------------------------------
# Judge (local)
# ---------------------------------------------------------------------------
def judge(
    config: ExperimentConfig,
    task: GenerationTask,
    questions: Sequence[Question],
    *,
    judge_impl: Judge,
    skip_uncached: bool = False,
) -> None:
    """Re-score one task's cached predictions (no GPU, no table build).

    `skip_uncached` (wired to `--continue-on-error`) skips cells that generate
    never produced instead of erroring, so a partial cache still scores what it
    has.
    """

    paths = experiment_paths(config, task.name)
    prediction_cache = PredictionCache(paths.predictions)
    result_cache = ResultCache(paths.results)
    guards = guard_retrievers()
    task_questions = task.resolve_questions(config, questions)

    specs = task.model_specs(config)
    if specs and len(prediction_cache) == 0:
        raise SystemExit(
            f"{task.name}: no cached predictions at {paths.predictions}; run the "
            "generate phase first (python -m cli.generate --generation ...)"
        )

    skipped = 0
    for spec in specs:
        orchestrator = Orchestrator(
            config,
            reasoner=_SpecOnlyReasoner(spec),
            judge=judge_impl,
            cache=result_cache,
            prediction_cache=prediction_cache,
        )
        for cell in task.generation_cells(config, task_questions, retrievers=guards):
            try:
                orchestrator.run_cell(cell.question, cell.conditioner, cell.representation)
            except CacheMiss:
                if not skip_uncached:
                    raise
                skipped += 1
    if skipped:
        log.warning(
            "[judge] %s: skipped %d uncached cell(s) (generate them for a complete table)",
            task.name,
            skipped,
        )


def run_judge(
    config: ExperimentConfig,
    selector: str,
    questions: Sequence[Question],
    *,
    judge_impl: Judge | None = None,
    continue_on_error: bool = False,
) -> list:
    """Score one task or a group; return per-task statuses. Builds no tables."""

    judge_impl = judge_impl or get_judge("gemini")
    statuses = []
    for task in resolve(selector):
        try:
            judge(config, task, questions, judge_impl=judge_impl, skip_uncached=continue_on_error)
        except Exception as exc:
            status = write_phase_status(config, task.name, phase="judge", status="failed", error=exc)
            statuses.append(status)
            log.error(
                "[judge] %s: FAILED (%s: %s)\n%s",
                task.name,
                status.error_type,
                status.error,
                "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            )
            if not continue_on_error:
                raise
            continue
        status = write_phase_status(config, task.name, phase="judge", status="success")
        statuses.append(status)
        log.info("[judge] %s: scored -> %s", task.name, experiment_paths(config, task.name).results)
    return statuses


# ---------------------------------------------------------------------------
# Shared CLI helper
# ---------------------------------------------------------------------------
def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Build the shared ExperimentConfig from parsed generate/judge/build args."""

    overrides: dict = {"smoke": not args.full, "sample": getattr(args, "questions", None)}
    if getattr(args, "per_bin_questions", None) is not None:
        overrides["per_bin_sample"] = args.per_bin_questions or None
    if getattr(args, "sample_seed", None) is not None:
        overrides["sample_seed"] = args.sample_seed
    if getattr(args, "quantization", None) is not None:
        overrides["quantization"] = args.quantization
    if getattr(args, "visual_resolution", None) is not None:
        overrides["visual_resolution"] = args.visual_resolution
    if getattr(args, "run_tag", None) is not None:
        overrides["run_tag"] = args.run_tag
    return ExperimentConfig(**overrides)
