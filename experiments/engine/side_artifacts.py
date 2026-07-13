"""Shared writers for per-run side artifacts."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from config import ExperimentConfig, max_pixels_for_resolution
from schema import Question

log = logging.getLogger("mpvrdu.side_artifacts")

# The RQ2 accuracy ladder defaults: text and vision cost rungs (cheap -> expensive)
# plus the matched-tier joint unions (pivot 4, 4.1). A spec overrides these via the
# G2 `retrieval` block; no reasoner touches them.
_TEXT_METHODS = ("bm25", "bge-m3", "qwen3-embedding")
_VISION_METHODS = ("colmodernvbert", "colqwen2.5", "colqwen3")
_JOINT_PAIRS = (("bm25", "colmodernvbert"), ("bge-m3", "colqwen2.5"), ("qwen3-embedding", "colqwen3"))


def resolve_joints(joints, text_methods: Sequence[str], vision_methods: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """Turn a `joints` spec into explicit (text, vision) pairs.

    `"matched"` auto-pairs the two method lists by cost-rung position (cheap|cheap,
    mid|mid, expensive|expensive). An explicit list of pairs is used as-is; `()`/None
    skips joints.
    """

    if joints == "matched":
        return tuple(zip(tuple(text_methods), tuple(vision_methods)))
    if not joints:
        return ()
    return tuple((str(t), str(v)) for t, v in joints)


def _build_retriever(config: ExperimentConfig, name: str, kind: str):
    """A fresh inner retriever with fallbacks off, so a model failure is an honest
    miss (caught per method) rather than a silent text/order ranking."""

    from retrievers.text import get_text_retriever
    from retrievers.vision import get_vision_retriever

    kwargs = dict(data_dir=config.paths.data_dir, cache_dir=config.paths.cache_dir, dpi=config.dpi)
    if kind == "text":
        if name != "bm25":
            kwargs["allow_bm25_fallback"] = False
        return get_text_retriever(name, **kwargs)
    kwargs["allow_text_fallback"] = False
    return get_vision_retriever(name, **kwargs)


def write_retrieval_eval(
    config: ExperimentConfig,
    questions: Sequence[Question],
    side_dir: Path,
    *,
    single_ks: Sequence[int],
    joint_ks: Sequence[int] = (1, 3, 5),
    text_methods: Sequence[str] = _TEXT_METHODS,
    vision_methods: Sequence[str] = _VISION_METHODS,
    joint_pairs: Sequence[tuple[str, str]] = _JOINT_PAIRS,
    filename: str = "retrieval.jsonl",
    fresh: bool = False,
) -> None:
    """Write the RQ2 retrieval-accuracy ladder (page P/R/F1 + cost), no reasoner.

    Scores every method in `text_methods` + `vision_methods` at `single_ks`, plus the
    `joint_pairs` unions at `joint_ks`, one row per (question, method, k). Each row
    carries the per-query `retrieval_latency_s` and the method's amortized
    `index_build_amortized_s`.

    Failures are isolated the way `predictions.jsonl` isolates cells: a method that
    fails to *load* is skipped whole, but once loaded each question is ranked
    independently, so a single OOM (a dense page on a V100) skips only that question
    and the rest keep going. A failed question is recorded in the memo as a status row
    (`status` + `skipped_reason`, no ranking) rather than silently dropped, and is left
    out of the scored benchmark rows. `fresh=True` deletes each method's memo file
    first, so the whole rung re-ranks under the current settings (no mixing, e.g.,
    capped and uncapped rows).
    """

    from data.loader import resolve_pdf
    from data.render import pdf_page_count
    from experiments.engine.driver import classify_failure
    from experiments.engine.paths import free_gpu
    from retrievers import MemoizedRetriever
    from retrievers.joint import union
    from scoring.retrieval import score_retrieval

    text_methods = tuple(text_methods)
    vision_methods = tuple(vision_methods)
    joint_pairs = tuple(joint_pairs)
    single_ks = tuple(int(k) for k in single_ks)
    joint_ks = tuple(int(k) for k in joint_ks)
    page_counts = {q.id: pdf_page_count(resolve_pdf(q.doc_id, config.paths.data_dir)) for q in questions}

    persist_dir = config.paths.cache_dir / "retrieval"
    # Rank each method over the corpus once. Wrapping in MemoizedRetriever persists
    # the full ranking to the shared retrieval memo (the same file build_retrievers
    # reads), so the inference stage reuses these rankings instead of ranking again.
    # A method that fails to load is simply absent (its rows and any joint using it
    # skip). Cost telemetry comes from the inner retriever.
    rankings: dict[str, dict[str, tuple[int, ...]]] = {}
    latency: dict[str, dict[str, float]] = {}
    index_build: dict[str, float] = {}
    modalities = {**{n: "text" for n in text_methods}, **{n: "vision" for n in vision_methods}}

    def _emit(handle, row) -> None:
        record = asdict(row)
        for key, value in list(record.items()):
            if isinstance(value, tuple):
                record[key] = list(value)
        handle.write(json.dumps(record, sort_keys=True) + "\n")

    side_dir.mkdir(parents=True, exist_ok=True)
    with (side_dir / filename).open("w") as handle:
        # Rank + score each single method and write its rows (then flush) as it
        # finishes, so a crash keeps every completed method's rows.
        for name, kind in [*((n, "text") for n in text_methods), *((n, "vision") for n in vision_methods)]:
            if fresh:
                (persist_dir / f"{name}__dpi{int(config.dpi)}.jsonl").unlink(missing_ok=True)
            try:
                retriever = MemoizedRetriever(_build_retriever(config, name, kind), persist_dir=persist_dir)
            except Exception as exc:  # noqa: BLE001 - a model that will not load skips the whole method
                log.warning("retrieval eval: method %s failed to build, skipping its rows: %s", name, exc)
                free_gpu()
                continue
            rmap: dict[str, tuple[int, ...]] = {}
            lmap: dict[str, float] = {}
            for q in questions:
                try:
                    rmap[q.id] = tuple(retriever.rank(q, page_counts[q.id]))
                    lmap[q.id] = float(getattr(retriever.inner, "last_query_s", 0.0))
                except Exception as exc:  # noqa: BLE001 - one question's OOM must not sink the method
                    status, reason = classify_failure(exc)
                    log.warning("retrieval eval: %s failed on question %s (%s), recording it: %s",
                                name, q.id, status, reason)
                    retriever.persist_failure(q.id, page_counts[q.id], status, reason)
                    free_gpu()
            idx = float(getattr(retriever.inner, "index_build_s", 0.0))
            rankings[name] = rmap
            latency[name] = lmap
            index_build[name] = idx
            for q in questions:
                if q.id not in rmap:
                    continue
                for k in single_ks:
                    _emit(handle, score_retrieval(
                        q, rmap[q.id][:k], retriever=name, modality=modalities[name], k=k,
                        retrieval_latency_s=lmap[q.id], index_build_amortized_s=idx, dpi=int(config.dpi)))
            handle.flush()
            retriever.unload()
            free_gpu()

        # Joint unions: emitted once both constituent methods have ranked (a question
        # missing from either side, e.g. one it OOM'd on, is skipped for the joint too).
        for tname, vname in joint_pairs:
            if tname not in rankings or vname not in rankings:
                continue
            idx = index_build.get(tname, 0.0) + index_build.get(vname, 0.0)
            joint_name = f"{tname}|{vname}"
            for q in questions:
                if q.id not in rankings[tname] or q.id not in rankings[vname]:
                    continue
                lat = latency[tname].get(q.id, 0.0) + latency[vname].get(q.id, 0.0)
                for k in joint_ks:
                    merged = union(rankings[tname][q.id][:k], rankings[vname][q.id][:k])
                    _emit(handle, score_retrieval(
                        q, merged, retriever=joint_name, modality="joint", k=k,
                        retrieval_latency_s=lat, index_build_amortized_s=idx, dpi=int(config.dpi)))
            handle.flush()


def write_classifier_eval(
    config: ExperimentConfig,
    questions: Sequence[Question],
    side_dir: Path,
    *,
    filename: str = "classifier.jsonl",
) -> None:
    """Classify each distinct document once and log predicted vs gold bin + latency."""

    from models.classifier import QwenBinClassifier

    classifier = QwenBinClassifier(
        data_dir=config.paths.data_dir,
        cache_dir=config.paths.cache_dir,
        dpi=config.dpi,
        max_pixels=max_pixels_for_resolution(config),
    )
    seen: set[str] = set()
    side_dir.mkdir(parents=True, exist_ok=True)
    with (side_dir / filename).open("w") as handle:
        for question in questions:
            if question.doc_id in seen:
                continue
            seen.add(question.doc_id)
            prediction = classifier.classify(question)
            gold_bin = question.bin_label
            predicted_bin = str(prediction.bin or gold_bin)
            handle.write(
                json.dumps(
                    {
                        "doc_id": question.doc_id,
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
