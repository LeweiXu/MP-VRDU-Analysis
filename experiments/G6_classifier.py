"""G6 classifier: the doc-type classifier over each document (side only).

Purpose:
    Runs the few-shot doc-type classifier once per document and logs predicted vs
    gold bin plus its latency. No reasoner cells: Table 7's routing policies reuse
    G1's ladder rows for accuracy and price predicted routing from this log.

Pipeline role:
    One `GenerationTask` with only `run_side` (the classifier is the sole GPU
    work). Builds Table 7 alongside G1's rows.

Arguments:
    None. Import-only; the registry instantiates `G6Classifier()`.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.base import GenerationTask


class G6Classifier(GenerationTask):
    name = "G6_classifier"
    side_artifact = "classifier.jsonl"

    def run_side(self, config, questions, side_dir: Path) -> None:
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
