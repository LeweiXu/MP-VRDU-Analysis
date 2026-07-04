"""Define the document-type classifier interface for routing policies.

Purpose:
    A `DocTypeClassifier` predicts a document's native type or bin so routing can
    choose a representation recipe when labels are unavailable. Its latency and
    accuracy are part of the RQ3 routing analysis.

Pipeline role:
    Policy runners call `classify(question)` and compare the prediction with the
    gold `doc_type` / Option-A bin. `StubClassifier` echoes the gold label until
    a cheap model classifier lands behind the same signature.

Arguments:
    None. This module is import-only; callers instantiate classifiers and call
    `classify()`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from schema import Question


@dataclass(frozen=True)
class DocTypePrediction:
    """A predicted document type plus the classifier's confidence."""

    doc_type: str
    confidence: float = 1.0


class DocTypeClassifier(ABC):
    """Predict a document's type for representation routing."""

    name: str = "classifier"

    @abstractmethod
    def classify(self, question: Question) -> DocTypePrediction:
        """Return the predicted document type and confidence for one question."""


class StubClassifier(DocTypeClassifier):
    """Deterministic placeholder: echo the gold `doc_type` at full confidence."""

    name = "stub"

    def classify(self, question: Question) -> DocTypePrediction:
        return DocTypePrediction(doc_type=question.doc_type, confidence=1.0)
