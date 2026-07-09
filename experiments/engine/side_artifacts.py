"""Shared writers for per-run side artifacts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from config import ExperimentConfig, max_pixels_for_resolution
from schema import Question


def write_retrieval_eval(
    config: ExperimentConfig,
    questions: Sequence[Question],
    pairs: Sequence[tuple[str, int]],
    side_dir: Path,
    *,
    filename: str = "retrieval.jsonl",
) -> None:
    """Write one page-retrieval R/P/F1 record per (question, modality, k).

    `pairs` is the ordered set of (retriever modality, k) to score. Duplicate
    pairs are dropped, first occurrence wins. A no-op if `pairs` is empty. This
    artifact is the retrieval-accuracy benchmark and never touches the reasoner.
    """

    from data.loader import resolve_pdf
    from data.render import pdf_page_count
    from retrievers import MemoizedRetriever
    from retrievers.text import Bm25Retriever
    from retrievers.vision import ColQwen25Retriever
    from scoring.retrieval import score_retrieval

    ordered: list[tuple[str, int]] = []
    seen_pairs: set[tuple[str, int]] = set()
    for modality, k in pairs:
        key = (str(modality), int(k))
        if key not in seen_pairs:
            seen_pairs.add(key)
            ordered.append(key)
    if not ordered:
        return

    persist_dir = config.paths.cache_dir / "retrieval"
    retrievers = {
        "text": MemoizedRetriever(
            Bm25Retriever(data_dir=config.paths.data_dir, cache_dir=config.paths.cache_dir, dpi=config.dpi),
            persist_dir=persist_dir,
        ),
        "vision": MemoizedRetriever(
            ColQwen25Retriever(data_dir=config.paths.data_dir, cache_dir=config.paths.cache_dir, dpi=config.dpi),
            persist_dir=persist_dir,
        ),
    }
    side_dir.mkdir(parents=True, exist_ok=True)
    with (side_dir / filename).open("w") as handle:
        for question in questions:
            page_count = pdf_page_count(resolve_pdf(question.doc_id, config.paths.data_dir))
            for modality, k in ordered:
                retriever = retrievers[modality]
                ranked = retriever.retrieve(question, page_count, k)
                record = asdict(
                    score_retrieval(question, ranked, retriever=retriever.name, modality=modality, k=k)
                )
                for key, value in list(record.items()):
                    if isinstance(value, tuple):
                        record[key] = list(value)
                handle.write(json.dumps(record, sort_keys=True) + "\n")


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
