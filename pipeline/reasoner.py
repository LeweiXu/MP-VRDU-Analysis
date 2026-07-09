"""Reasoner ABC: answer a Question given a ModelInput."""

from __future__ import annotations

from abc import ABC, abstractmethod

from models.payload import ModelInput
from schema import Prediction, Question


class Reasoner(ABC):
    """Answer a question given the document represented as a `ModelInput`."""

    #: The model spec string this reasoner serves (recorded in the cell key).
    spec: str = "reasoner"
    #: Instruction preamble for the next answer; the orchestrator sets this per
    #: cell from the cell's prompt mode. None means the backend's default.
    prompt_instruction: str | None = None

    @abstractmethod
    def answer(self, question: Question, model_input: ModelInput) -> Prediction:
        """Return a `Prediction` for the question over the given model input."""


class StubReasoner(Reasoner):
    """Deterministic placeholder that returns a fixed answer with zeroed cost."""

    def __init__(self, spec: str = "stub") -> None:
        self.spec = spec

    def answer(self, question: Question, model_input: ModelInput) -> Prediction:
        return Prediction(
            text="stub-answer",
            model_spec=self.spec,
            metadata={
                "n_text_parts": len(model_input.text_parts),
                "n_image_parts": len(model_input.image_parts),
            },
        )
