"""Backend-agnostic reasoner interface for answering questions from model inputs.

Stage C of the pipeline. A `Reasoner` takes a `Question` and the document's
`ModelInput` (the representation, adapted for a backend) and returns a
`Prediction`. This ABC is the swap point: the pipeline asks the `models/`
registry for a `Reasoner` and never imports a concrete backend, so substituting
Qwen3-VL sizes, other open families, or a closed API is a registry change, not a
pipeline change.

Stage 3 ships only `StubReasoner`, which returns a fixed answer with zeroed cost
fields. The real local (vLLM/HF) and API backends land in Stage 6 behind this
same `answer(question, model_input)` signature.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from models.payload import ModelInput
from schema import Prediction, Question


class Reasoner(ABC):
    """Answer a question given the document represented as a `ModelInput`."""

    #: The model spec string this reasoner serves (recorded in the cache key).
    spec: str = "reasoner"

    @abstractmethod
    def answer(self, question: Question, model_input: ModelInput) -> Prediction:
        """Return a `Prediction` for the question over the given model input."""


class StubReasoner(Reasoner):
    """Deterministic placeholder used until real backends arrive in Stage 6."""

    def __init__(self, spec: str = "stub") -> None:
        self.spec = spec

    def answer(self, question: Question, model_input: ModelInput) -> Prediction:
        n_text = len(model_input.text_parts)
        n_image = len(model_input.image_parts)
        return Prediction(
            text="stub-answer",
            model_spec=self.spec,
            metadata={"n_text_parts": n_text, "n_image_parts": n_image},
        )
