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

from pathlib import Path

from experiments.base import GenerationTask
from experiments.side_artifacts import write_classifier_eval


class G6Classifier(GenerationTask):
    name = "G6_classifier"
    side_artifact = "classifier.jsonl"

    def run_side(self, config, questions, side_dir: Path) -> None:
        """Classify each distinct document once and log bin/latency."""

        write_classifier_eval(config, questions, side_dir, filename=self.side_artifact)
