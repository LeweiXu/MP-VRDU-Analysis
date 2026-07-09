"""Cell-level run primitives: read cell rows, run every cell to exactly one row
regardless of outcome, select the failed rows, and merge a failed-only re-run in
place."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any


def read_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield each jsonl row of a predictions/results file as a dict."""

    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _field(item: Any, name: str, default: Any = None) -> Any:
    """Read a field from a mapping row/cell or a dataclass-like object."""

    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def classify_failure(exc: BaseException) -> tuple[str, str]:
    """Map an exception to a `(status, skipped_reason)` pair.

    A CUDA out-of-memory reads as `oom` (the expected, recoverable failure a
    failed-only re-run on a bigger GPU completes); anything else is `error`.
    """

    reason = f"{type(exc).__name__}: {exc}"
    status = "oom" if "out of memory" in str(exc).lower() else "error"
    return status, reason


def _failed_row(cell: Any, exc: BaseException) -> dict[str, Any]:
    """Default failure row: carry the cell's identity, stamp status + reason."""

    status, reason = classify_failure(exc)
    row = dict(cell) if isinstance(cell, Mapping) else {"prediction_key": _field(cell, "prediction_key")}
    row["status"] = status
    row["skipped_reason"] = reason
    row["oom_occurred"] = status == "oom"
    return row


def run_cells(
    cells: Sequence[Any],
    run_one: Callable[[Any], Any],
    *,
    on_failure: Callable[[Any, BaseException], Any] | None = None,
) -> list[Any]:
    """Run `run_one` over every cell and return one row per cell.

    A cell that succeeds contributes `run_one(cell)`; a cell that raises
    contributes a failure row (via `on_failure`, default `_failed_row`) carrying
    its identity plus `status` in {oom, error} and a `skipped_reason`. The output
    always has exactly `len(cells)` rows, so a failure is data, never a hole.
    """

    build_failed = on_failure or _failed_row
    rows: list[Any] = []
    for cell in cells:
        try:
            rows.append(run_one(cell))
        except Exception as exc:  # noqa: BLE001 - a cell failure is recorded, not raised
            rows.append(build_failed(cell, exc))
    return rows


def select_failed(rows: Sequence[Any]) -> list[Any]:
    """Return the rows whose status is not `ok` (the re-run work queue)."""

    return [row for row in rows if _field(row, "status") != "ok"]


def merge_failed_only(existing: Sequence[Any], reruns: Sequence[Any]) -> list[Any]:
    """Upgrade failed rows in place from a failed-only re-run.

    A row keyed by `prediction_key` that was not `ok` and appears in `reruns` is
    replaced by its re-run; `ok` rows are left untouched. The result has the same
    rows as `existing` (no duplicates), converging the file toward complete.
    """

    reruns_by_key = {_field(r, "prediction_key"): r for r in reruns}
    merged: list[Any] = []
    for row in existing:
        key = _field(row, "prediction_key")
        if _field(row, "status") != "ok" and key in reruns_by_key:
            merged.append(reruns_by_key[key])
        else:
            merged.append(row)
    return merged


# -- the generate/judge run loop --------------------------------------------

import logging  # noqa: E402

log = logging.getLogger("mpvrdu.driver")


def build_retrievers(config):
    """The text + vision retrievers a run may need, memoized and lazily loaded."""

    from experiments.tasks.base import Retrievers
    from retrievers import MemoizedRetriever
    from retrievers.text import Bm25Retriever
    from retrievers.vision import ColQwen25Retriever

    persist_dir = config.paths.cache_dir / "retrieval"
    return Retrievers(
        text=MemoizedRetriever(
            Bm25Retriever(data_dir=config.paths.data_dir, cache_dir=config.paths.cache_dir, dpi=config.dpi),
            persist_dir=persist_dir,
        ),
        vision=MemoizedRetriever(
            ColQwen25Retriever(data_dir=config.paths.data_dir, cache_dir=config.paths.cache_dir, dpi=config.dpi),
            persist_dir=persist_dir,
        ),
    )


def _modality_of(cell) -> str:
    """The ladder rung a cell targets, whether it holds a str or a composer."""

    rep = cell.representation
    return rep if isinstance(rep, str) else rep.modality


def _warm_parser_cache(config, pages) -> None:
    """Warm parser markdown for the given TL/TLV pages in one isolated-env pass.

    The parser runs in its own env and its output crosses to the reasoner only
    through the disk cache, so this happens in the pre-pass with no reasoner
    loaded. `pages` is already deduplicated by the caller. A parser that cannot
    run is logged, not raised: its TL/TLV cells then record a parser-miss row
    rather than sinking the whole task.
    """

    if not pages:
        return
    from tools.parser import warm_parser_cache

    try:
        warm_parser_cache(pages, parser_tool=config.parser_tool, dpi=config.dpi)
        log.info("parser warm: %s over %d pages", config.parser_tool, len(pages))
    except Exception as exc:  # noqa: BLE001 - a cold parser cache is a cell miss, not a task crash
        log.warning("parser warm (%s) failed over %d pages: %s", config.parser_tool, len(pages), exc)


def _failed_result_row(orchestrator, cell, exc, machine):
    """Build a status row for a cell that raised, so it is data, not a hole."""

    from pipeline.representation import get_representation
    from schema import ResultRow

    rep = get_representation(cell.representation) if isinstance(cell.representation, str) else cell.representation
    status, reason = classify_failure(exc)
    try:
        page_set = cell.conditioner.condition(cell.question, orchestrator.page_count(cell.question))
        pages, provenance, note = page_set.page_indices, page_set.provenance, page_set.note
    except Exception:
        pages, provenance, note = (), "", ""
    prediction_key, result_key = orchestrator._keys(cell.question, cell.conditioner, rep.modality, pages)
    q = cell.question
    return ResultRow(
        result_key=result_key, prediction_key=prediction_key, question_id=q.id, doc_id=q.doc_id,
        doc_type=q.doc_type, bin_label=q.bin_label, scan_label=q.scan_label, hop=q.hop,
        is_unanswerable=q.is_unanswerable, evidence_sources=q.evidence_sources,
        condition=cell.conditioner.name, provenance=provenance, page_indices=pages,
        representation=rep.modality, model_spec=orchestrator.reasoner.spec, judge_spec=orchestrator.judge.spec,
        machine=machine, status=status, skipped_reason=reason, oom_occurred=(status == "oom"),
        answer="", score=0.0, correct=False, abstained=False, total_text_tokens=0, total_visual_tokens=0,
        text_tokens_fed=0, output_tokens=0, tokens_dropped=0, truncation_occurred=False,
        latency_s=0.0, prefill_latency_s=0.0, decode_latency_s=0.0, peak_vram_bytes=0, note=note, metadata={},
    )


def generate(config, task, questions, *, limit=None, machine=None):
    """Run one task's cells to a scored row each, then its side work.

    A parse pre-pass warms the retrieval and render caches with the reasoner not
    yet loaded, then the retriever weights are freed before the reasoner loads, so
    parser/retriever/reasoner never share the GPU. Every cell writes exactly one
    row (ok, or a status row on failure) via `run_cells`.
    """

    from config import max_pixels_for_resolution
    from experiments.engine.paths import experiment_paths, free_gpu
    from models import get_reasoner
    from pipeline.judge import StubJudge, get_judge
    from pipeline.orchestrator import Orchestrator, PredictionCache, ResultCache, current_machine
    from pipeline.reasoner import Reasoner

    machine = machine or current_machine()
    paths = experiment_paths(config, task.name)
    result_cache = ResultCache(paths.results)
    prediction_cache = PredictionCache(paths.predictions)
    retrievers = build_retrievers(config)
    task_questions = list(task.resolve_questions(config, questions))
    if limit is not None:
        task_questions = task_questions[:limit]
    specs = task.model_specs(config)
    log.info("generate %s | %d questions | specs=%s", task.name, len(task_questions), list(specs) or "(side-only)")

    class _SpecOnly(Reasoner):
        def __init__(self, spec):
            self.spec = spec

        def answer(self, question, model_input):
            raise RuntimeError("spec-only reasoner must not run inference")

    for spec in specs:
        cells = task.generation_cells(config, task_questions, retrievers=retrievers)
        prewarm = Orchestrator(config, reasoner=_SpecOnly(spec), judge=StubJudge("prewarm"),
                               cache=result_cache, prediction_cache=prediction_cache)
        parser_pages, seen_pages = [], set()
        for cell in cells:
            try:
                page_set = prewarm.prewarm_cell(cell.question, cell.conditioner, cell.representation)
            except Exception as exc:  # noqa: BLE001 - warming is best effort
                log.warning("prewarm failed q=%s: %s", cell.question.id, exc)
                continue
            if _modality_of(cell) not in ("TL", "TLV"):
                continue
            for page in prewarm.render_pages(cell.question, page_set):
                key = (str(page.pdf_path), page.index)
                if key not in seen_pages:
                    seen_pages.add(key)
                    parser_pages.append(page)
        _warm_parser_cache(config, parser_pages)
        retrievers.text.unload()
        retrievers.vision.unload()
        free_gpu()

        reasoner = get_reasoner(spec, max_pixels=max_pixels_for_resolution(config), max_new_tokens=config.max_tokens)
        orchestrator = Orchestrator(config, reasoner=reasoner, judge=get_judge(config.judge_spec),
                                    cache=result_cache, prediction_cache=prediction_cache, machine=machine)

        def run_one(cell):
            return orchestrator.run_cell(cell.question, cell.conditioner, cell.representation, cell.prompt_mode)

        def on_failure(cell, exc):
            row = _failed_result_row(orchestrator, cell, exc, machine)
            result_cache.put(row)
            log.error("cell FAILED q=%s rep=%s: %s", cell.question.id, cell.representation, row.skipped_reason)
            return row

        run_cells(cells, run_one, on_failure=on_failure)
        if hasattr(reasoner, "free"):
            reasoner.free()
        free_gpu()

    log.info("generate %s: side work", task.name)
    task.run_side(config, task_questions, paths.side_dir)
    free_gpu()
