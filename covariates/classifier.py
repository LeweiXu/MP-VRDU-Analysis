"""Document-type classifier interface for representation-routing covariates.

A `DocTypeClassifier` is the RQ3 routing covariate: a cheap pass that predicts a
document's type so the runner can apply that type's representation frontier
instead of one uniform representation. Its accuracy vs the gold `doc_type` bounds
whether routing can pay off, so it is logged alongside routed accuracy.

Stage 3 ships only `StubClassifier` (echoes the gold label at full confidence).
The real cheap-model classifier lands in Stage 8 behind this same
`classify(question)` signature.
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
