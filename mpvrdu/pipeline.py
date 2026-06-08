"""Pipeline: config -> data slice -> select evidence -> generate -> score -> JSONL.

One run = one config. This is the spine every sub-study plugs into; only the
config changes between conditions. Each question emits one JSONL row with the
prediction, the per-question scores, and the retrieval recall, so any run is
fully auditable and the analysis stage (Stage 7) reads only JSONL.

    python -m mpvrdu.pipeline --config configs/smoke.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

from .config import RunConfig, load_config
from .data.dataset import Dataset
from .data.load import load_dataset
from .eval.judge import build_judge
from .eval.metrics import aggregate, recall_at_k
from .generate.base import InputBuilder
from .generate.factory import build_generator
from .generate.mock import MockGenerator
from .logging_utils import get_logger, set_seed
from .results import ResultsWriter, results_path
from .retrieve import build_selector

log = get_logger("pipeline")


def run(cfg: RunConfig, dataset: Optional[Dataset] = None,
        out_path: Optional[str | Path] = None) -> dict:
    """Execute one run. Returns the aggregate metrics dict.

    Two-phase to keep peak GPU memory at one model (fits 12GB): phase 1 computes
    every question's retrieval selection (visual-retriever model resident), then
    frees that model; phase 2 loads the generator and answers. Phase 1 visits
    questions in document order so the per-document index is reused.
    """
    set_seed(cfg.seed)
    log.info("run %s (hash %s)", cfg.name, cfg.hash())

    if dataset is None:
        dataset = load_dataset(cfg.data)
    log.info("dataset: %d questions, %d docs, types=%s",
             len(dataset), len(dataset.documents), dataset.type_counts())

    selector = build_selector(cfg)
    generator = build_generator(cfg.generation, dataset)
    judge = build_judge(cfg.judge)
    builder = InputBuilder(cfg)

    out_path = Path(out_path) if out_path else results_path(cfg)
    t0 = time.time()

    # --- phase 1: retrieval selections (document-sorted for index reuse) ---
    questions = dataset.questions
    doc_order = sorted(range(len(questions)), key=lambda i: questions[i].doc_id)
    selections: dict[str, object] = {}
    for i in doc_order:
        q = questions[i]
        selections[q.qid] = selector.select(q, dataset.get_document(q.doc_id))
    selector.unload()      # free any visual-retriever GPU model before generation

    # --- phase 2: generation + scoring (original order) ---
    scores = []
    recalls = []
    with ResultsWriter(out_path, config=cfg) as writer:
        for q in questions:
            doc = dataset.get_document(q.doc_id)
            selection = selections[q.qid]

            # mock(gold) needs to know which question it is answering
            if isinstance(generator, MockGenerator):
                generator.set_qid(q.qid)

            gi = builder.build(q, doc, selection)
            pred = generator.answer_input(gi)

            ascore = judge.score(q.question, pred, q.answer, q.answer_format,
                                 gold_is_unanswerable=q.is_unanswerable)
            recall = recall_at_k(selection.page_indices,
                                 q.evidence_pages_zero_based, k=cfg.retrieval.top_k)
            scores.append(ascore)
            recalls.append(recall)

            writer.write({
                "qid": q.qid,
                "doc_id": q.doc_id,
                "question": q.question,
                "question_type": q.question_type.value,
                "answer_format": q.answer_format,
                "gold": q.answer,
                "pred": pred,
                "correct": ascore.correct,
                "pred_abstained": ascore.pred_abstained,
                "gold_answerable": ascore.gold_answerable,
                "evidence_pages": q.evidence_pages,
                "selected_pages_0based": selection.page_indices,
                "n_selected": gi.n_pages,
                "recall_at_k": recall,
                "evidence_sources": q.evidence_sources,
            })

    metrics = aggregate(scores)
    metrics["mean_recall_at_k"] = sum(recalls) / max(len(recalls), 1)
    metrics["seconds"] = round(time.time() - t0, 2)
    metrics["results_path"] = str(out_path)
    log.info("DONE %s | acc=%.3f f1=%.3f recall@%d=%.3f n=%d (%.1fs) -> %s",
             cfg.name, metrics["accuracy"], metrics["f1"], cfg.retrieval.top_k,
             metrics["mean_recall_at_k"], metrics["n"], metrics["seconds"], out_path)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one MP-VRDU pipeline config.")
    ap.add_argument("--config", required=True, help="path to a run YAML config")
    ap.add_argument("--out", default=None, help="override output JSONL path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run(cfg, out_path=args.out)


if __name__ == "__main__":
    main()
