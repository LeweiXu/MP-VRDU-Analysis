# kaya: target=gpu
# kaya: env=true
# kaya: offline=true
# kaya: job-name=generate
"""Generation tasks: the only experiment work that needs a GPU.

Purpose:
    Organizes the study by *what gets generated*, not by paper table. A
    `GenerationTask` declares the reasoner specs, conditioned cells, corpus, and
    any GPU side work (retrieval diagnostics, the doc-type classifier) it needs.
    Most tables are pure aggregations of these caches, so a handful of tasks
    covers all eight tables:

    - G1_sufficiency : oracle pages x the T/TL/TLV/V ladder, primary reasoner.
      The source rows for tables 1, 2, 5, 7.
    - G2_family      : the same ladder on a second family (InternVL3-8B). Table 3.
    - G3_dataset     : the same ladder on a held-out MMLongBench document subset
      (text_heavy + in_between only). Table 4.
    - G5_retrieval   : matched/cross retrieval cells + retrieval R/P/F1. Table 6.
    - G6_classifier  : the doc-type classifier over each document (side only, no
      reasoner cells). Table 7's predicted-routing price.

    (A scale-sanity task for 2B/32B, feeding Table 8, is out of scope for now.)

Pipeline role:
    This module is the GPU half. It caches predictions per task under
    `results/cache/<mode>/<task>/`; `experiments/judge.py` scores them and
    `experiments/build.py` aggregates across tasks into the table CSVs. Submitted
    to Kaya with `kaya.kaya submit experiments/generation.py -- --generation ...`.

CLI:
    `python -m experiments.generation [--generation SEL] [--full] [options]`

Arguments:
    --generation SEL: a task name (`G1_sufficiency`), a group (`all`,
        `reasoners`), or a comma list. --full uses the 8B/full corpus (smoke
        otherwise). --quantization / --visual-resolution / --run-tag tune the
        reasoner and cache namespace; --continue-on-error keeps a grouped run
        going after a task fails. See `build_parser` for the rest.
"""

from __future__ import annotations

import argparse
import os

# Reduce CUDA fragmentation on the compute nodes; the allocator reads this on the
# first CUDA alloc, so setting it before generation runs is enough. setdefault
# lets a submit script override it. (The real vision-token fix is the per-page
# pixel cap / --visual-resolution.)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from config import ExperimentConfig, max_input_tokens_for_spec, max_pixels_for_resolution
from covariates.retriever import BM25BGERetriever, ColQwenRetriever, MemoizedRetriever, Retriever
from experiments.corpus import load_questions, sample_table4_replication
from experiments.paths import (
    answer_preview,
    configure_logging,
    experiment_paths,
    free_gpu,
    log,
    mode,
    write_phase_status,
)
from pipeline.conditioner import InputConditioner, OracleConditioner, RetrievedTopK
from pipeline.orchestrator import Orchestrator, PredictionCache, ResultCache
from pipeline.judge import StubJudge
from schema import Modality, Question
from scripts.prestage import prepare_tool_cache_env


# ---------------------------------------------------------------------------
# Cell factories (the reasoner work shapes shared across tasks)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Cell:
    """One unit of reasoner work: a question under a condition and a rung."""

    question: Question
    conditioner: InputConditioner
    representation: Modality


@dataclass(frozen=True)
class Retrievers:
    """The two retrievers a generation pass may need (built lazily by the driver).

    The generate phase passes real retrievers; the judge phase passes guards
    that raise if called (every retrieved cell must be a prediction-cache hit).
    """

    text: Retriever
    vision: Retriever


def oracle_ladder_cells(config: ExperimentConfig, questions: Sequence[Question]) -> list[Cell]:
    """Oracle pages x the full representation ladder (the sufficiency cells)."""

    oracle = OracleConditioner()
    return [Cell(question, oracle, rung) for question in questions for rung in config.representations]


def matched_cross_cells(
    questions: Sequence[Question],
    *,
    retrievers: Retrievers,
    k: int,
    representation: Modality = "TLV",
) -> list[Cell]:
    """Matched (vision-retrieval) and cross (text-retrieval) vision-reasoning cells.

    Conditioner names are fixed strings so the prediction key is identical across
    the generate and judge phases regardless of which retriever object is passed.
    """

    vision = RetrievedTopK(retrievers.vision, k, name=f"retrieved_vision_k{k}")
    text = RetrievedTopK(retrievers.text, k, name=f"retrieved_text_k{k}")
    cells: list[Cell] = []
    for question in questions:
        cells.append(Cell(question, vision, representation))
        cells.append(Cell(question, text, representation))
    return cells


# ---------------------------------------------------------------------------
# The GenerationTask contract and the concrete tasks
# ---------------------------------------------------------------------------
class GenerationTask(ABC):
    """One coarse generation pass: specs + cells + optional side work."""

    #: Stable short name; also the cache subdirectory (e.g. "G1_sufficiency").
    name: str = "generation"
    #: Side-artifact filename this task writes into its cache dir, if any.
    side_artifact: str | None = None

    def model_specs(self, config: ExperimentConfig) -> tuple[str, ...]:
        """Reasoner specs this task generates. Empty for side-only tasks."""

        return ()

    def resolve_questions(
        self, config: ExperimentConfig, questions: Sequence[Question]
    ) -> Sequence[Question]:
        """The corpus this task runs on (defaults to the shared corpus)."""

        return questions

    def generation_cells(
        self, config: ExperimentConfig, questions: Sequence[Question], *, retrievers: Retrievers
    ) -> list[Cell]:
        """Reasoner cells to run per spec. Empty for side-only tasks."""

        return []

    def run_side(
        self, config: ExperimentConfig, questions: Sequence[Question], side_dir: Path
    ) -> None:
        """Optional extra GPU work (retrieval diagnostics, classifier)."""

        return None


class G1Sufficiency(GenerationTask):
    """Oracle ladder for the primary reasoner: the core sufficiency generation."""

    name = "G1_sufficiency"

    def model_specs(self, config: ExperimentConfig) -> tuple[str, ...]:
        return (config.reasoner_spec,)

    def generation_cells(self, config, questions, *, retrievers) -> list[Cell]:
        return oracle_ladder_cells(config, questions)


class G2Family(GenerationTask):
    """Oracle ladder on a second model family (InternVL3-8B). Table 3 replication."""

    name = "G2_family"

    def model_specs(self, config: ExperimentConfig) -> tuple[str, ...]:
        # Smoke has one family (Table 3 reuses G1). Full adds the second family.
        return () if config.smoke else ("internvl3-8b-local",)

    def generation_cells(self, config, questions, *, retrievers) -> list[Cell]:
        return oracle_ladder_cells(config, questions)


class G3Dataset(GenerationTask):
    """Oracle ladder on a held-out MMLongBench subset (text_heavy + in_between)."""

    name = "G3_dataset"

    def model_specs(self, config: ExperimentConfig) -> tuple[str, ...]:
        return () if config.smoke else (config.reasoner_spec,)

    def resolve_questions(self, config, questions) -> Sequence[Question]:
        if config.smoke:
            return questions
        from data.loader import load_mmlongbench

        all_questions = list(load_mmlongbench(data_dir=config.paths.data_dir))
        # Held out only for the two big bins; visual_heavy is too thin to hold out
        # and is out of scope here (no reuse, no SlideVQA yet).
        held_out_bins = tuple(b for b in config.bins if b != "visual_heavy")
        return sample_table4_replication(
            all_questions,
            config.per_bin_sample or 100,
            bins=held_out_bins,
            seed=config.sample_seed,
            reuse_bins=(),
        )

    def generation_cells(self, config, questions, *, retrievers) -> list[Cell]:
        return oracle_ladder_cells(config, questions)


class G5Retrieval(GenerationTask):
    """Matched vs cross retrieval cells + retrieval R/P/F1 side records. Table 6."""

    name = "G5_retrieval"
    side_artifact = "retrieval.jsonl"

    def _top_k(self, config: ExperimentConfig) -> int:
        return int(config.k_values[0] if config.k_values else 1)

    def model_specs(self, config: ExperimentConfig) -> tuple[str, ...]:
        return (config.reasoner_spec,)

    def generation_cells(self, config, questions, *, retrievers) -> list[Cell]:
        return matched_cross_cells(questions, retrievers=retrievers, k=self._top_k(config))

    def run_side(self, config, questions, side_dir) -> None:
        """Log page R/P/F1 for both retrievers (evidence-modality diagnostic)."""

        from dataclasses import asdict

        from data.loader import resolve_pdf
        from data.render import pdf_page_count
        from metrics.retrieval import score_retrieval

        top_k = self._top_k(config)
        text = MemoizedRetriever(
            BM25BGERetriever(data_dir=config.paths.data_dir, cache_dir=config.paths.cache_dir, dpi=config.dpi)
        )
        vision = MemoizedRetriever(
            ColQwenRetriever(data_dir=config.paths.data_dir, cache_dir=config.paths.cache_dir, dpi=config.dpi)
        )
        side_dir.mkdir(parents=True, exist_ok=True)
        with (side_dir / self.side_artifact).open("w") as handle:
            for question in questions:
                page_count = pdf_page_count(resolve_pdf(question.doc_id, config.paths.data_dir))
                for modality, retriever in (("vision", vision), ("text", text)):
                    ranked = retriever.retrieve(question, page_count, top_k)
                    record = asdict(
                        score_retrieval(question, ranked, retriever=retriever.name, modality=modality, k=top_k)
                    )
                    for key, value in list(record.items()):
                        if isinstance(value, tuple):
                            record[key] = list(value)
                    handle.write(json.dumps(record, sort_keys=True) + "\n")


class G6Classifier(GenerationTask):
    """Doc-type classifier over each document (side only). Table 7 routing price."""

    name = "G6_classifier"
    side_artifact = "classifier.jsonl"

    def run_side(self, config, questions, side_dir) -> None:
        """Classify each distinct document once and log bin/latency."""

        from covariates.classifier import QwenDocTypeClassifier
        from data.binning import doc_type_bin

        classifier = QwenDocTypeClassifier(
            data_dir=config.paths.data_dir,
            cache_dir=config.paths.cache_dir,
            dpi=config.dpi,
            max_pixels=config.max_pixels,
            max_input_tokens=config.max_input_tokens,
        )
        seen: set[str] = set()
        side_dir.mkdir(parents=True, exist_ok=True)
        with (side_dir / self.side_artifact).open("w") as handle:
            for question in questions:
                if question.doc_id in seen:
                    continue
                seen.add(question.doc_id)
                prediction = classifier.classify(question)
                gold_bin = doc_type_bin(question.doc_type)
                predicted_bin = str(prediction.bin or gold_bin)
                handle.write(
                    json.dumps(
                        {
                            "doc_id": question.doc_id,
                            "gold_doc_type": question.doc_type,
                            "predicted_doc_type": prediction.doc_type,
                            "gold_bin": gold_bin,
                            "predicted_bin": predicted_bin,
                            "correct_bin": predicted_bin == gold_bin,
                            "confidence": prediction.confidence,
                            "latency_s": prediction.latency_s,
                            "classifier": classifier.name,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_TASKS: tuple[GenerationTask, ...] = (
    G1Sufficiency(),
    G2Family(),
    G3Dataset(),
    G5Retrieval(),
    G6Classifier(),
)

GENERATION_TASKS: dict[str, GenerationTask] = {task.name: task for task in _TASKS}
ORDER: tuple[str, ...] = tuple(task.name for task in _TASKS)

GROUPS: dict[str, tuple[str, ...]] = {
    "all": ORDER,
    # Just the reasoner-cell tasks (skip the classifier side task).
    "reasoners": ("G1_sufficiency", "G2_family", "G3_dataset", "G5_retrieval"),
}


def resolve(selector: str) -> list[GenerationTask]:
    """Expand a selector to generation tasks, in registry order, de-duplicated.

    A selector is a task name (`G1_sufficiency`), a group (`all`, `reasoners`),
    or a comma-separated list of either, so an ad-hoc subset runs as one job.
    """

    names: list[str] = []
    for token in selector.split(","):
        key = token.strip()
        if not key:
            continue
        if key in GROUPS:
            names.extend(GROUPS[key])
        elif key in GENERATION_TASKS:
            names.append(key)
        else:
            raise ValueError(
                f"unknown generation task/group {key!r}; choose from "
                f"{sorted(GENERATION_TASKS)} or groups {sorted(GROUPS)}"
            )
    ordered = [name for name in ORDER if name in set(names)]
    return [GENERATION_TASKS[name] for name in ordered]


# ---------------------------------------------------------------------------
# GPU-side helpers
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


def generate(config: ExperimentConfig, task: GenerationTask, questions: Sequence[Question]) -> None:
    """Phase 1 (GPU): cache every prediction for one task, then run side work."""

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
            except Exception:
                log.error("FAILED %s (after %.1fs)", label, time.perf_counter() - started)
                raise
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

        # Release the reasoner before the next spec or the side work, so a
        # multi-spec task (or the classifier in run_side) starts from a clean GPU.
        if hasattr(reasoner, "free"):
            reasoner.free()
        del orchestrator
        reasoner = None
        free_gpu()

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
            generate(config, task, questions)
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
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", default="all", help="generation task or group (default: all)")
    parser.add_argument("--full", action="store_true", help="use the full config/corpus (default: smoke)")
    parser.add_argument("--questions", type=int, help="global cap: first N questions (overrides --per-bin-questions)")
    parser.add_argument("--per-bin-questions", type=int, help="full mmlongbench: ~N questions per Option-A bin by whole documents (default 100; 0 = whole corpus)")
    parser.add_argument("--sample-seed", type=int, help="which documents land in the per-bin subset (default 0)")
    parser.add_argument("--quantization", choices=("4bit", "8bit"), help="load the local reasoner quantized so 8B fits one 16GB V100 (bf16 by default)")
    parser.add_argument("--visual-resolution", choices=("full", "high", "med", "low", "min"), help="fix the per-page vision-token budget (overrides the size-aware default; lower = more downscaling)")
    parser.add_argument("--run-tag", help="namespace this run's cache tree (results/cache/<TAG>/) so parallel full runs don't share files; judge/build with the same tag")
    parser.add_argument("--continue-on-error", action="store_true", help="continue after a task failure and record its status")
    parser.add_argument("--verbose", action="store_true", help="DEBUG-level per-cell/per-stage logging (smoke runs are verbose by default)")
    parser.add_argument("--quiet", action="store_true", help="force INFO-level logging even for smoke runs")
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Build the shared ExperimentConfig from parsed generate/judge args."""

    overrides: dict = {"smoke": not args.full, "sample": args.questions}
    if args.per_bin_questions is not None:
        overrides["per_bin_sample"] = args.per_bin_questions or None
    if args.sample_seed is not None:
        overrides["sample_seed"] = args.sample_seed
    if args.quantization is not None:
        overrides["quantization"] = args.quantization
    if getattr(args, "visual_resolution", None) is not None:
        overrides["visual_resolution"] = args.visual_resolution
    if args.run_tag is not None:
        overrides["run_tag"] = args.run_tag
    return ExperimentConfig(**overrides)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    # Smoke runs are verbose by default (they exist to surface failures); --quiet
    # opts out, --verbose forces DEBUG for a full run too.
    configure_logging(verbose=args.verbose or (config.smoke and not args.quiet))
    prepare_tool_cache_env(config.paths.hf_home)
    questions = load_questions(config, limit=args.questions)
    statuses = run_generate(config, args.generation, questions, continue_on_error=args.continue_on_error)
    failed = [status for status in statuses if status.status != "success"]
    print(
        f"generated {args.generation}: {len(questions)} questions "
        f"({'full' if args.full else 'smoke'}), {len(statuses) - len(failed)} succeeded, {len(failed)} failed"
    )
    for status in failed:
        print(f"failed {status.experiment}: {status.error_type}: {status.error} ({status.path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
