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
    """The text + vision retrievers the inference stage feeds the reasoner, memoized
    and lazily loaded. Which arm is which comes from the spec
    (`inference_text_retriever` / `inference_vision_retriever`, default bm25 /
    colqwen2.5); the full six-method benchmark lives in the retrieval side-artifact.
    """

    from experiments.tasks.base import Retrievers
    from retrievers import MemoizedRetriever, StubRetriever
    from retrievers.text import get_text_retriever
    from retrievers.vision import get_vision_retriever

    persist_dir = config.paths.cache_dir / "retrieval"
    kwargs = dict(data_dir=config.paths.data_dir, cache_dir=config.paths.cache_dir, dpi=config.dpi)

    def _text_arm():
        # A run with no text arm (oracle-only, e.g. G1) gets a stub that is never fed.
        if str(config.inference_text_retriever).lower() in ("none", ""):
            return StubRetriever()
        text_kwargs = dict(kwargs)
        if config.inference_text_retriever != "bm25":
            text_kwargs["allow_bm25_fallback"] = False
        return get_text_retriever(config.inference_text_retriever, **text_kwargs)

    def _vision_arm():
        if str(config.inference_vision_retriever).lower() in ("none", ""):
            return StubRetriever()
        return get_vision_retriever(config.inference_vision_retriever, allow_text_fallback=False, **kwargs)

    return Retrievers(
        text=MemoizedRetriever(_text_arm(), persist_dir=persist_dir),
        vision=MemoizedRetriever(_vision_arm(), persist_dir=persist_dir),
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


def _warm_parser_after_retrieval(config, pages, retrievers, free_gpu) -> None:
    """Unload retrieval models before the isolated parser process takes the GPU."""

    retrievers.text.unload()
    retrievers.vision.unload()
    free_gpu()
    _warm_parser_cache(config, pages)
    free_gpu()


def _failed_prediction_row(orchestrator, cell, exc, machine):
    """Build a status `PredictionRow` for a cell that raised, so it is data, not a hole."""

    from pipeline.representation import get_representation
    from schema import PredictionRow

    rep = get_representation(cell.representation) if isinstance(cell.representation, str) else cell.representation
    status, reason = classify_failure(exc)
    try:
        page_set = cell.conditioner.condition(cell.question, orchestrator.page_count(cell.question))
        pages, provenance, note = page_set.page_indices, page_set.provenance, page_set.note
    except Exception:
        pages, provenance, note = (), "", ""
    prediction_key = orchestrator._prediction_key(cell.question, cell.conditioner, rep.modality, pages)
    q = cell.question
    return PredictionRow(
        prediction_key=prediction_key, question_id=q.id, doc_id=q.doc_id,
        doc_type=q.doc_type, bin_label=q.bin_label, scan_label=q.scan_label, hop=q.hop,
        is_unanswerable=q.is_unanswerable, evidence_sources=q.evidence_sources,
        condition=cell.conditioner.name, provenance=provenance, page_indices=pages,
        representation=rep.modality, model_spec=orchestrator.reasoner.spec,
        machine=machine, status=status, skipped_reason=reason, oom_occurred=(status == "oom"),
        answer="", total_text_tokens=0, total_visual_tokens=0,
        text_tokens_fed=0, output_tokens=0, tokens_dropped=0, truncation_occurred=False,
        latency_s=0.0, prefill_latency_s=0.0, decode_latency_s=0.0, peak_vram_bytes=0,
        visual_resolution=orchestrator.visual_resolution, note=note, metadata={},
    )


def _cell_identity(cell, spec, resolution) -> tuple:
    """The (question, doc, condition, rung, model, resolution) a cell and its row share."""

    q = cell.question
    return (q.id, q.doc_id, cell.conditioner.name, _modality_of(cell), spec, resolution)


def _prepare_failed_only(predictions_path) -> set[tuple]:
    """Drop failed rows from a predictions file and return the cells to re-run.

    Reads the existing rows, keeps the `ok` ones, rewrites the file with just
    those, and returns the identity of each dropped (failed) cell. Removing the
    failed rows is what lets the re-run recompute them: a cell whose error row is
    still cached would otherwise read straight back as a cache hit.
    """

    path = Path(predictions_path)
    if not path.exists():
        return set()
    ok_rows: list[dict] = []
    failed: set[tuple] = set()
    for row in read_rows(path):
        if (row.get("status") or "ok") == "ok":
            ok_rows.append(row)
        else:
            failed.add((row.get("question_id"), row.get("doc_id"), row.get("condition"),
                        row.get("representation"), row.get("model_spec"),
                        row.get("visual_resolution", "")))
    if not failed:
        return set()
    with path.open("w") as handle:
        for row in ok_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return failed


def _oom_cell_ids(predictions_path) -> set[tuple]:
    """Identities of cells already recorded as `oom` in a predictions file.

    Unlike `_prepare_failed_only`, this does not touch the file: a cached oom row
    is a cache hit and would be skipped at inference anyway, but `--skip-oom` also
    drops these cells from the prewarm / parser-warm pass, which is where a resume
    otherwise still pays to render + parse pages for cells that only OOM again.
    """

    path = Path(predictions_path)
    if not path.exists():
        return set()
    oom: set[tuple] = set()
    for row in read_rows(path):
        if (row.get("status") or "ok") == "oom":
            oom.add((row.get("question_id"), row.get("doc_id"), row.get("condition"),
                     row.get("representation"), row.get("model_spec"),
                     row.get("visual_resolution", "")))
    return oom


def generate(config, task, questions, *, limit=None, machine=None, failed_only=False,
             skip_retrieval=False, skip_oom=False):
    """Run one task's cells to an unjudged prediction row each, then its side work.

    A parse pre-pass warms the retrieval and render caches with the reasoner not
    yet loaded, then the retriever weights are freed before the reasoner loads, so
    parser/retriever/reasoner never share the GPU. Every cell writes exactly one
    `PredictionRow` (ok, or a status row on failure) to `predictions.jsonl` via
    `run_cells`; scoring is a separate judge phase.

    `skip_retrieval` skips the stage-1 retrieval benchmark on a normal run (not just a
    failed-only retry), so a resumed / supervisor inference pass reuses the existing
    retrieval memo and never rewrites `retrieval.jsonl`. It needs the memo already
    present (else inference re-ranks by loading the retrievers).

    `skip_oom` drops every cell already recorded as `oom` in `predictions.jsonl` from
    the run (prewarm + parser-warm included), so a V100 resume does not re-render and
    re-parse pages for cells that will only OOM again. Leave those cells for a
    `--failed-only` sweep on a bigger GPU. It is the resume counterpart to
    `--failed-only` (which retries them); passing both together is contradictory.
    """

    from config import VISUAL_RESOLUTION_PRESETS
    from experiments.engine.paths import experiment_paths, free_gpu
    from models import get_reasoner
    from pipeline.orchestrator import Orchestrator, PredictionCache, current_machine
    from pipeline.reasoner import Reasoner

    machine = machine or current_machine()
    paths = experiment_paths(config, task.name)
    failed_ids: set[tuple] | None = None
    if failed_only:
        failed_ids = _prepare_failed_only(paths.predictions)
        if not failed_ids:
            log.info("failed-only %s: no failed cells, nothing to retry", task.name)
            return
        log.info("failed-only %s: retrying %d failed cells", task.name, len(failed_ids))
    oom_ids: set[tuple] = _oom_cell_ids(paths.predictions) if skip_oom else set()
    if skip_oom:
        log.info("skip-oom %s: dropping %d cells already recorded as oom", task.name, len(oom_ids))
    prediction_cache = PredictionCache(paths.predictions)
    task_questions = list(task.resolve_questions(config, questions))
    if limit is not None:
        task_questions = task_questions[:limit]
    specs = task.model_specs(config)
    resolutions = config.visual_resolutions or (config.visual_resolution,)
    log.info("generate %s | %d questions | specs=%s | resolutions=%s",
             task.name, len(task_questions), list(specs) or "(side-only)", list(resolutions))

    # The retrieval-accuracy benchmark is stage 1: it ranks every method and persists
    # the rankings to the shared retrieval memo, which the inference retrievers below
    # reuse instead of ranking the shared methods a second time. So run it before the
    # reasoner cells (GPU free, no reasoner resident). On a failed-only retry the memo
    # already exists from the first run, so skip it. A run with configured benchmark
    # method lists is the one that has this stage; others just have post-loop side work.
    # `--skip-retrieval` forces the reuse path even on a normal run.
    retrieval_stage_first = bool(config.text_retrievers) and not failed_only and not skip_retrieval
    if retrieval_stage_first:
        log.info("generate %s: retrieval stage-1 (before inference)", task.name)
        task.run_retrieval_benchmark(config, questions, paths.side_dir, limit=limit)
        free_gpu()
    elif skip_retrieval and bool(config.text_retrievers):
        memo = config.paths.cache_dir / "retrieval"
        if memo.exists() and any(memo.glob("*.jsonl")):
            log.info("generate %s: --skip-retrieval, reusing memo at %s (stage-1 skipped, retrieval.jsonl untouched)",
                     task.name, memo)
        else:
            log.warning("generate %s: --skip-retrieval but no retrieval memo at %s; "
                        "inference will re-rank (load retrievers)", task.name, memo)

    retrievers = build_retrievers(config)

    class _SpecOnly(Reasoner):
        def __init__(self, spec):
            self.spec = spec

        def answer(self, question, model_input):
            raise RuntimeError("spec-only reasoner must not run inference")

    for spec in specs:
        base_cells = task.generation_cells(config, task_questions, retrievers=retrievers)
        if failed_ids is not None:
            base_cells = [c for c in base_cells
                          if any(_cell_identity(c, spec, r) in failed_ids for r in resolutions)]
            if not base_cells:
                log.info("failed-only %s: spec %s has no failed cells, skipping", task.name, spec)
                continue
        if oom_ids:
            # Keep a cell in the prewarm scope if at least one resolution is not oom;
            # the per-resolution loop below drops the individual oom resolutions.
            base_cells = [c for c in base_cells
                          if not all(_cell_identity(c, spec, r) in oom_ids for r in resolutions)]
            if not base_cells:
                log.info("skip-oom %s: spec %s is entirely oom cells, skipping", task.name, spec)
                continue

        # Pre-pass (resolution-independent): warm the render / retrieval / parser
        # caches once, with no reasoner resident. Resolution is a reasoner-side
        # downscale, so it does not change what gets rendered or parsed here.
        prewarm = Orchestrator(config, reasoner=_SpecOnly(spec), prediction_cache=prediction_cache)
        parser_pages, seen_pages = [], set()
        for cell in base_cells:
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
        _warm_parser_after_retrieval(config, parser_pages, retrievers, free_gpu)

        # Reasoner loads once per spec; each resolution just changes the per-page
        # vision-token budget (a processor-side downscale), so we reuse the loaded
        # weights across resolutions rather than reloading them.
        reasoner = get_reasoner(spec, max_pixels=VISUAL_RESOLUTION_PRESETS[resolutions[0]],
                                max_new_tokens=config.max_tokens)
        for resolution in resolutions:
            cells = base_cells
            if failed_ids is not None:
                cells = [c for c in base_cells if _cell_identity(c, spec, resolution) in failed_ids]
                if not cells:
                    continue
            if oom_ids:
                cells = [c for c in cells if _cell_identity(c, spec, resolution) not in oom_ids]
                if not cells:
                    continue
            reasoner.max_pixels = VISUAL_RESOLUTION_PRESETS[resolution]
            orchestrator = Orchestrator(config, reasoner=reasoner, prediction_cache=prediction_cache,
                                        machine=machine, visual_resolution=resolution)

            def run_one(cell, _orch=orchestrator):
                return _orch.run_cell(cell.question, cell.conditioner, cell.representation, cell.prompt_mode)

            def on_failure(cell, exc, _orch=orchestrator):
                row = _failed_prediction_row(_orch, cell, exc, machine)
                prediction_cache.put(row)
                log.error("cell FAILED q=%s rep=%s res=%s: %s",
                          cell.question.id, cell.representation, _orch.visual_resolution, row.skipped_reason)
                return row

            run_cells(cells, run_one, on_failure=on_failure)

        if hasattr(reasoner, "free"):
            reasoner.free()
        free_gpu()

    if failed_only:
        # A failed-only re-run retries reasoner cells; side artifacts (which have
        # no per-cell status) are regenerated wholesale on a normal run.
        return
    log.info("generate %s: side work", task.name)
    # Side writers get the full corpus (not the task pool) and the smoke limit, so a
    # writer whose scope differs from the task's cells (e.g. G3's classifier, which
    # prices G1's answerable docs while G3's cells run the unanswerable pool) can
    # resolve its own set. Each writer re-filters to keep its scope correct.
    task.run_side(config, questions, paths.side_dir, limit=limit)
    free_gpu()
