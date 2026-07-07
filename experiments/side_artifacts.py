"""Shared side-artifact writers for retrieval and classifier diagnostics.

Purpose:
    The retrieval R/P/F1 log and the doc-type classifier log are each produced by
    two entry points: the fixed `G5Retrieval` / `G6Classifier` tasks and the
    dynamic YAML tasks (`experiments/yaml_spec.py`). This module holds the single
    implementation both call, so the two paths cannot silently drift.

Pipeline role:
    A leaf helper used only inside a task's `run_side`. The caller decides *which*
    units to score (the (modality, k) pairs for retrieval); this module does the
    scoring and writes the JSONL. It never touches the reasoner or the cache keys.

Arguments:
    None. Import-only. Public entry points: `write_retrieval_eval`,
    `write_classifier_eval`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from config import ExperimentConfig
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

    `pairs` is the ordered set of (retriever modality, k) to score; the caller
    decides which pairs matter and in what order they are emitted (G5 scores both
    modalities across the full k-sweep; a YAML run scores only the (retriever, k)
    present in its retrieved conditions). Duplicate pairs are dropped, first
    occurrence wins, so the caller's order is preserved. A no-op if `pairs` is
    empty.
    """

    from covariates.retriever import BM25BGERetriever, ColQwenRetriever, MemoizedRetriever
    from data.loader import resolve_pdf
    from data.render import pdf_page_count
    from metrics.retrieval import score_retrieval

    ordered: list[tuple[str, int]] = []
    seen_pairs: set[tuple[str, int]] = set()
    for modality, k in pairs:
        key = (str(modality), int(k))
        if key not in seen_pairs:
            seen_pairs.add(key)
            ordered.append(key)
    if not ordered:
        return

    retrievers = {
        "text": MemoizedRetriever(
            BM25BGERetriever(data_dir=config.paths.data_dir, cache_dir=config.paths.cache_dir, dpi=config.dpi)
        ),
        "vision": MemoizedRetriever(
            ColQwenRetriever(data_dir=config.paths.data_dir, cache_dir=config.paths.cache_dir, dpi=config.dpi)
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
    with (side_dir / filename).open("w") as handle:
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
