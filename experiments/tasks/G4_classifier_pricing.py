"""Prices the document classifier's latency and VRAM; emits no reasoner cells."""

from __future__ import annotations

from pathlib import Path

from experiments.engine.side_artifacts import write_classifier_eval
from experiments.tasks.base import GenerationTask


class G4ClassifierPricing(GenerationTask):
    name = "G4_classifier_pricing"
    side_artifact = "classifier.jsonl"

    def resolve_questions(self, config, questions):
        # The classifier prices over documents, not answerable-only questions, so
        # it uses the whole corpus rather than a task pool.
        return questions

    def run_side(self, config, questions, side_dir: Path) -> None:
        """Classify each distinct document once and log bin/latency."""

        write_classifier_eval(config, questions, side_dir, filename=self.side_artifact)
