"""Score model predictions against gold answers through a uniform interface.

Purpose:
    Defines Stage D of the pipeline. A `Judge` converts a `Prediction` and
    `Question` into a comparable `Score`, keeping answer evaluation independent
    of representation and reasoner backend.

Pipeline role:
    The orchestrator applies one judge implementation across all cells so table
    columns are commensurable. The real judge should be a *different model
    family* from the reasoner (Qwen3-VL) so it is an independent scorer for the
    κ-validation gate. Two API judges are provided behind the same interface:
    `GPT4oMiniJudge` (OpenAI, paid) and `GeminiJudge` (Google, has a free tier).
    `StubJudge` stays for offline tests and smoke cache plumbing.

Arguments:
    None. This module is import-only; callers instantiate a `Judge` subclass or
    call `get_judge(spec)`.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any

from metrics.abstention import is_abstention
from schema import Prediction, Question, Score


class Judge(ABC):
    """Score a prediction against a question's gold answer."""

    spec: str = "judge"

    @abstractmethod
    def score(self, question: Question, prediction: Prediction) -> Score:
        """Return a `Score` for the prediction on this question."""


class StubJudge(Judge):
    """Heuristic placeholder judge used until the real judge arrives in Stage 7."""

    def __init__(self, spec: str = "stub") -> None:
        self.spec = spec

    def score(self, question: Question, prediction: Prediction) -> Score:
        abstained = is_abstention(prediction.text)
        gold = question.gold_answer.strip().casefold()
        pred = prediction.text.strip().casefold()
        if question.is_unanswerable:
            correct = abstained
        else:
            correct = bool(gold) and gold in pred and not abstained
        return Score(
            value=1.0 if correct else 0.0,
            correct=correct,
            abstained=abstained,
            judge_spec=self.spec,
        )


JUDGE_SYSTEM_PROMPT = """You judge answers to document questions.
Return only JSON with keys:
- verdict: one of correct, incorrect, abstained
- extracted_answer: the answer extracted from the model response, or empty string
- rationale: a short reason

Mark correct when the model answer is semantically equivalent to the gold answer.
For unanswerable questions, mark correct only when the model abstains.
"""


def _response_text(response: Any) -> str:
    """Extract text from OpenAI chat-completion-like response objects."""

    choices = response["choices"] if isinstance(response, dict) else response.choices
    first = choices[0]
    message = first["message"] if isinstance(first, dict) else first.message
    content = message["content"] if isinstance(message, dict) else message.content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(getattr(item, "text", "")))
        return "\n".join(part for part in parts if part)
    return str(content)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object from a model response."""

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match is None:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError(f"judge response JSON must be an object, got {type(value).__name__}")
    return value


def _judge_user_payload(question: Question, prediction: Prediction) -> str:
    """Serialise the fields an LLM judge scores, stable across backends."""

    return json.dumps(
        {
            "question": question.question,
            "gold_answer": question.gold_answer,
            "is_unanswerable": question.is_unanswerable,
            "model_answer": prediction.text,
        },
        sort_keys=True,
    )


def _score_from_verdict(
    question: Question,
    prediction: Prediction,
    payload: dict[str, Any],
    *,
    judge_spec: str,
    model: str,
) -> Score:
    """Turn a parsed judge JSON verdict into the common `Score` contract."""

    verdict = str(payload.get("verdict", "")).strip().casefold()
    abstained = verdict == "abstained" or is_abstention(prediction.text)
    correct = verdict == "correct"
    if question.is_unanswerable and abstained:
        correct = True
    return Score(
        value=1.0 if correct else 0.0,
        correct=correct,
        abstained=abstained,
        judge_spec=judge_spec,
        metadata={
            "verdict": verdict,
            "extracted_answer": str(payload.get("extracted_answer", "")),
            "rationale": str(payload.get("rationale", "")),
            "model": model,
        },
    )


class GPT4oMiniJudge(Judge):
    """OpenAI GPT-4o-mini judge that returns the common `Score` contract."""

    spec = "gpt4o-mini-judge"

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        client: Any | None = None,
        spec: str | None = None,
    ) -> None:
        self.model = model
        self.spec = spec or self.spec
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client = client

    def score(self, question: Question, prediction: Prediction) -> Score:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": _judge_user_payload(question, prediction)},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        payload = _extract_json_object(_response_text(response))
        return _score_from_verdict(question, prediction, payload, judge_spec=self.spec, model=self.model)


class GeminiJudge(Judge):
    """Google Gemini judge (different family, free tier) via the google-genai SDK.

    Reads `GEMINI_API_KEY` (or `GOOGLE_API_KEY`). `google-genai` is already a
    dependency (Marker pulls it in), so this adds no new package. Flash models
    have a free tier, which is why this is a good default for the smoke run and
    the κ-validation gate.
    """

    spec = "gemini-flash-judge"

    def __init__(
        self,
        *,
        model: str = "gemini-2.5-flash",
        client: Any | None = None,
        spec: str | None = None,
    ) -> None:
        self.model = model
        self.spec = spec or self.spec
        if client is None:
            from google import genai

            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.client = client

    def score(self, question: Question, prediction: Prediction) -> Score:
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model,
            contents=_judge_user_payload(question, prediction),
            config=types.GenerateContentConfig(
                system_instruction=JUDGE_SYSTEM_PROMPT,
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        payload = _extract_json_object(response.text or "")
        return _score_from_verdict(question, prediction, payload, judge_spec=self.spec, model=self.model)


def get_judge(spec: str) -> Judge:
    """Return a judge implementation for a config spec."""

    normalized = spec.strip().casefold()
    if normalized in {"stub", ""}:
        return StubJudge("stub")
    if normalized in {"gpt4o-mini", "gpt-4o-mini", "gpt4o-mini-judge", "gpt-4o-mini-judge"}:
        return GPT4oMiniJudge(spec=spec)
    if normalized in {"gemini", "gemini-flash", "gemini-flash-judge", "gemini-2.5-flash"}:
        return GeminiJudge(spec=spec)
    raise ValueError(f"unsupported judge spec {spec!r}")
