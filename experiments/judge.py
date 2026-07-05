"""Judge phase: re-score cached predictions locally, no GPU.

Purpose:
    The middle role. It reads a generation task's cached predictions and scores
    each with a real judge (a different model family from the reasoner), writing
    judged rows to that task's `results.jsonl`. It builds no tables: aggregation
    is `experiments/build.py`'s job. Splitting it out is what lets the whole judge
    step run off the cluster on just an API key.

Pipeline role:
    Reads `results/cache/<mode>/<task>/predictions.jsonl` (written by
    `experiments/generation.py`), writes `results.jsonl` next to it. The judge
    phase must never run a reasoner or retriever: the guards below raise
    `CacheMiss` if a cell was never generated, which `--continue-on-error` turns
    into a skip so a partial cache still scores what it has.

CLI:
    `python -m experiments.judge [--generation SEL] [--full] [--judge SPEC]`

    Pass the same corpus/model flags as the generate phase (`--full`,
    `--per-bin-questions`, `--sample-seed`, `--quantization`, `--visual-resolution`,
    `--run-tag`) so the judge re-resolves the exact same cells.

Arguments:
    --generation SEL: which task(s) to score (name, group, or comma list).
        --judge picks the scorer (gemini default, gpt-4o-mini, stub).
        --continue-on-error skips cells with no cached prediction (partial cache)
        and continues past a task failure. The corpus/model flags must match the
        generate phase; see `build_parser`.
"""

from __future__ import annotations

import argparse
import traceback
from collections.abc import Sequence

from config import ExperimentConfig
from experiments.corpus import load_questions
from experiments.generation import GenerationTask, Retrievers, config_from_args, resolve
from experiments.paths import configure_logging, experiment_paths, log, write_phase_status
from pipeline.judge import Judge, get_judge
from pipeline.orchestrator import Orchestrator, PredictionCache, ResultCache
from pipeline.reasoner import Reasoner
from covariates.retriever import Retriever
from schema import Prediction, Question


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


def judge(
    config: ExperimentConfig,
    task: GenerationTask,
    questions: Sequence[Question],
    *,
    judge_impl: Judge,
    skip_uncached: bool = False,
) -> None:
    """Re-score one generation task's cached predictions (no GPU, no build).

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
            "generate phase first (python -m experiments.generation --generation ...)"
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", default="all", help="generation task or group to score (default: all)")
    parser.add_argument("--full", action="store_true", help="use the full config/corpus (default: smoke)")
    parser.add_argument("--judge", default="gemini", help="judge: gemini (default), gpt-4o-mini, or stub")
    parser.add_argument("--questions", type=int, help="global cap: first N questions (overrides --per-bin-questions)")
    parser.add_argument("--per-bin-questions", type=int, help="full mmlongbench: ~N questions per Option-A bin (default 100; 0 = whole corpus)")
    parser.add_argument("--sample-seed", type=int, help="which documents land in the per-bin subset (default 0)")
    parser.add_argument("--quantization", choices=("4bit", "8bit"), help="quantized reasoner spec suffix; must match the generate phase")
    parser.add_argument("--visual-resolution", choices=("full", "high", "med", "low", "min"), help="ignored for scoring; kept so judge flags can mirror generate")
    parser.add_argument("--run-tag", help="cache namespace to read; must match the generate phase")
    parser.add_argument("--continue-on-error", action="store_true", help="skip cells with no cached prediction (partial cache) and continue past task failures")
    parser.add_argument("--verbose", action="store_true", help="DEBUG-level logging (smoke runs are verbose by default)")
    parser.add_argument("--quiet", action="store_true", help="force INFO-level logging even for smoke runs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    configure_logging(verbose=args.verbose or (config.smoke and not args.quiet))
    questions = load_questions(config, limit=args.questions)
    statuses = run_judge(
        config,
        args.generation,
        questions,
        judge_impl=get_judge(args.judge),
        continue_on_error=args.continue_on_error,
    )
    failed = [status for status in statuses if status.status != "success"]
    print(
        f"judged {args.generation}: {len(statuses) - len(failed)} scored, {len(failed)} failed. "
        f"Build tables with: python -m experiments.build"
        f"{' --full' if args.full else ''}{f' --run-tag {args.run_tag}' if args.run_tag else ''}"
    )
    for status in failed:
        print(f"failed {status.experiment}: {status.error_type}: {status.error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
