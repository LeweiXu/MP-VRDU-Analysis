"""Experiment T7 — RQ3: routing under classification cost.

Purpose:
    Compares four policies (oracle routing, predicted routing, uniform-cheapest
    `T`, uniform-strongest `TLV`) on the accuracy–cost trade, with the doc-type
    classifier's own latency folded into predicted routing (Table 7). The policy
    rows are built from T1's oracle-ladder rows; the classifier is the only new
    GPU work and is run once per document as a side artifact.

Pipeline role:
    A concrete `Experiment` with `depends_on = ("T1_headline",)`, no reasoner
    cells, and a `run_side` that runs `QwenDocTypeClassifier` and logs predicted
    vs gold bin plus latency. `build` reads that log to price predicted routing.

Arguments:
    None. Import-only; the driver instantiates `Routing()` via the registry.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from config import ExperimentConfig
from data.binning import doc_type_bin
from experiments.base import Experiment, bootstrap_resamples
from experiments.tables import build_table7_routing
from pipeline.orchestrator import ResultRow


CLASSIFIER_LOG = "classifier.jsonl"


class Routing(Experiment):
    name = "T7_routing"
    tables = ("table7",)
    depends_on = ("T1_headline",)

    def run_side(self, config: ExperimentConfig, questions: Sequence, side_dir: Path) -> None:
        """Classify each distinct document once and log bin/latency."""

        from covariates.classifier import QwenDocTypeClassifier

        classifier = QwenDocTypeClassifier(
            data_dir=config.paths.data_dir,
            cache_dir=config.paths.cache_dir,
            dpi=config.dpi,
            max_pixels=config.max_pixels,
            max_input_tokens=config.max_input_tokens,
        )
        seen: set[str] = set()
        side_dir.mkdir(parents=True, exist_ok=True)
        with (side_dir / CLASSIFIER_LOG).open("w") as handle:
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

    def build(
        self, config: ExperimentConfig, rows: Sequence[ResultRow], side_dir: Path
    ) -> Mapping[str, pd.DataFrame]:
        table = build_table7_routing(
            rows,
            bins=config.bins,
            margin_points=config.sufficiency_margin,
            classifier_records=self._classifier_records(side_dir / CLASSIFIER_LOG),
            n_bootstrap=bootstrap_resamples(config),
        )
        return {"table7": table}

    @staticmethod
    def _classifier_records(log_path: Path) -> list[dict]:
        """Load classifier side records, returning empty list if absent."""

        if not log_path.exists():
            return []
        return [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
