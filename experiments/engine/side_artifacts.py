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

# The RQ2 accuracy ladder: text and vision cost rungs (cheap -> expensive) plus the
# three matched-tier joint unions (pivot 4, 4.1). No reasoner touches these.
_TEXT_METHODS = ("bm25", "bge-m3", "qwen3-embedding")
_VISION_METHODS = ("colmodernvbert", "colqwen2.5", "colqwen3")
_JOINT_PAIRS = (("bm25", "colmodernvbert"), ("bge-m3", "colqwen2.5"), ("qwen3-embedding", "colqwen3"))


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
    filename: str = "retrieval.jsonl",
) -> None:
    """Write the RQ2 retrieval-accuracy ladder (page P/R/F1 + cost), no reasoner.

    Scores all six methods (three text, three vision cost rungs) at `single_ks`,
    plus the three matched-tier joint unions at `joint_ks`, one row per (question,
    method, k). Each row carries the per-query `retrieval_latency_s` and the
    method's amortized `index_build_amortized_s`. Each method is built and scored
    independently so a big-model OOM (Qwen3-Embedding-4B, ColQwen3-4B on a V100)
    loses only that method's rows, not the whole artifact.
    """

    from data.loader import resolve_pdf
    from data.render import pdf_page_count
    from experiments.engine.paths import free_gpu
    from retrievers.joint import union
    from scoring.retrieval import score_retrieval

    single_ks = tuple(int(k) for k in single_ks)
    joint_ks = tuple(int(k) for k in joint_ks)
    page_counts = {q.id: pdf_page_count(resolve_pdf(q.doc_id, config.paths.data_dir)) for q in questions}

    # Rank every method the corpus once, keeping full rankings + timing. A method
    # that fails to load is simply absent (its rows and any joint using it skip).
    rankings: dict[str, dict[str, tuple[int, ...]]] = {}
    latency: dict[str, dict[str, float]] = {}
    index_build: dict[str, float] = {}
    modalities = {**{n: "text" for n in _TEXT_METHODS}, **{n: "vision" for n in _VISION_METHODS}}

    for name, kind in [*((n, "text") for n in _TEXT_METHODS), *((n, "vision") for n in _VISION_METHODS)]:
        try:
            retriever = _build_retriever(config, name, kind)
            rmap: dict[str, tuple[int, ...]] = {}
            lmap: dict[str, float] = {}
            for q in questions:
                rmap[q.id] = tuple(retriever.rank(q, page_counts[q.id]))
                lmap[q.id] = float(getattr(retriever, "last_query_s", 0.0))
            rankings[name] = rmap
            latency[name] = lmap
            index_build[name] = float(getattr(retriever, "index_build_s", 0.0))
            if hasattr(retriever, "unload"):
                retriever.unload()
        except Exception as exc:  # noqa: BLE001 - one method's failure must not sink the artifact
            log.warning("retrieval eval: method %s failed, skipping its rows: %s", name, exc)
        finally:
            free_gpu()

    def _emit(handle, row) -> None:
        record = asdict(row)
        for key, value in list(record.items()):
            if isinstance(value, tuple):
                record[key] = list(value)
        handle.write(json.dumps(record, sort_keys=True) + "\n")

    side_dir.mkdir(parents=True, exist_ok=True)
    with (side_dir / filename).open("w") as handle:
        for name in (*_TEXT_METHODS, *_VISION_METHODS):
            if name not in rankings:
                continue
            idx = index_build.get(name, 0.0)
            for q in questions:
                ranking = rankings[name].get(q.id, ())
                lat = latency[name].get(q.id, 0.0)
                for k in single_ks:
                    _emit(handle, score_retrieval(
                        q, ranking[:k], retriever=name, modality=modalities[name], k=k,
                        retrieval_latency_s=lat, index_build_amortized_s=idx))
        for tname, vname in _JOINT_PAIRS:
            if tname not in rankings or vname not in rankings:
                continue
            idx = index_build.get(tname, 0.0) + index_build.get(vname, 0.0)
            joint_name = f"{tname}|{vname}"
            for q in questions:
                lat = latency[tname].get(q.id, 0.0) + latency[vname].get(q.id, 0.0)
                for k in joint_ks:
                    merged = union(rankings[tname][q.id][:k], rankings[vname][q.id][:k])
                    _emit(handle, score_retrieval(
                        q, merged, retriever=joint_name, modality="joint", k=k,
                        retrieval_latency_s=lat, index_build_amortized_s=idx))


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
