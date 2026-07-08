"""Define the backend-agnostic reasoner contract.

Purpose:
    Defines Stage C of the pipeline. A `Reasoner` takes a `Question` plus a
    backend-neutral `ModelInput` and returns a `Prediction` with answer text and
    cost accounting.

Pipeline role:
    The orchestrator receives concrete reasoners from `models.get_reasoner()`
    but depends only on this ABC. That is the model-family swap point for
    Qwen3-VL, InternVL, local vLLM/HF backends, and future API backends.

Current implementation:
    `StubReasoner` keeps tests and cache plumbing runnable until real model
    backends land.

Arguments:
    None. This module is import-only; callers implement or instantiate
    `Reasoner` subclasses.
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
