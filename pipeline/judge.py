"""Judge interface and API-backed scorers."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar

from config import JUDGE_GEMINI_MODEL, JUDGE_GPT_MODEL, JUDGE_SYSTEM_PROMPT
from schema import Prediction, Question, Score
from scoring.abstention import is_abstention

log = logging.getLogger("mpvrdu.judge")

_T = TypeVar("_T")

# HTTP statuses worth another try: rate limits (429) and transient server errors
# (5xx). Free-tier judge endpoints return sporadic 503s and 429s; without a retry
# one blip kills a whole judge run mid-corpus even though the work so far is
# cached. Non-transient errors (400 bad request, 401 auth) raise on first attempt.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_RETRYABLE_NAMES = {"ServerError", "APIConnectionError", "APITimeoutError"}


def _is_daily_quota(exc: Exception) -> bool:
    """True for a per-day quota exhaustion (RPD), not a transient per-minute 429.

    A daily cap won't clear for ~a day, so retrying the same key is pointless; the
    caller should switch to a fallback key instead. Gemini spells the daily metric
    out in the error body ("per_model_per_day" / "PerDay").
    """

    msg = str(exc)
    return "per_model_per_day" in msg or "PerDay" in msg


def _is_quota_error(exc: Exception) -> bool:
    """True for any rate/quota rejection (429 or RESOURCE_EXHAUSTED)."""

    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int) and code == 429:
        return True
    return "RESOURCE_EXHAUSTED" in str(exc)


def _is_retryable(exc: Exception) -> bool:
    # A daily-quota 429 is not retryable on the same key; raise so a multi-key
    # judge can fall back to the next key at once.
    if _is_daily_quota(exc):
        return False
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int) and code in _RETRYABLE_STATUS:
        return True
    return type(exc).__name__ in _RETRYABLE_NAMES


def _with_retry(call: Callable[[], _T], *, attempts: int = 6, base_delay: float = 2.0) -> _T:
    """Call an API judge with exponential backoff on transient 429/5xx errors."""

    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:  # narrowed by _is_retryable; non-transient re-raises
            if attempt == attempts - 1 or not _is_retryable(exc):
                raise
            delay = base_delay * (2**attempt)
            log.warning("judge call failed (%s), retry %d/%d in %.0fs", exc, attempt + 1, attempts - 1, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")  # loop either returns or raises


class Judge(ABC):
    """Score a prediction against a question's gold answer."""

    spec: str = "judge"

    @abstractmethod
    def score(self, question: Question, prediction: Prediction) -> Score:
        """Return a `Score` for the prediction on this question."""


class StubJudge(Judge):
    """Heuristic placeholder judge for offline tests and cache plumbing."""

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

    def __init__(self, *, model: str = JUDGE_GPT_MODEL, client: Any | None = None, spec: str | None = None) -> None:
        self.model = model
        self.spec = spec or self.spec
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client = client

    def score(self, question: Question, prediction: Prediction) -> Score:
        response = _with_retry(
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": _judge_user_payload(question, prediction)},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
        )
        payload = _extract_json_object(_response_text(response))
        return _score_from_verdict(question, prediction, payload, judge_spec=self.spec, model=self.model)


def _gemini_api_keys() -> list[str]:
    """Ordered, de-duplicated Gemini API keys from the environment.

    Primary first (`GEMINI_API_KEY`, then `GOOGLE_API_KEY`), then the optional
    `GEMINI_API_KEY_SECONDARY` fallback. A run that exhausts the primary key's
    free-tier daily quota rolls over to the next key instead of failing.
    """

    keys: list[str] = []
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY_SECONDARY"):
        value = (os.environ.get(name) or "").strip()
        if value and value not in keys:
            keys.append(value)
    return keys


class GeminiJudge(Judge):
    """Google Gemini judge (different family, free tier) via the google-genai SDK.

    Reads `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) plus an optional
    `GEMINI_API_KEY_SECONDARY`. With more than one key it fails over to the next
    on a quota/rate error, which matters because the free tier caps both
    per-minute and per-day request counts.
    """

    spec = "gemini-flash-judge"

    def __init__(self, *, model: str = JUDGE_GEMINI_MODEL, client: Any | None = None, spec: str | None = None) -> None:
        self.model = model
        self.spec = spec or self.spec
        self._index = 0  # sticky: once a key is exhausted we stay on the fallback
        if client is not None:
            self._clients: list[Any] = [client]
            return
        from google import genai

        keys = _gemini_api_keys()
        self._clients = [genai.Client(api_key=key) for key in keys] if keys else [genai.Client()]

    def score(self, question: Question, prediction: Prediction) -> Score:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=JUDGE_SYSTEM_PROMPT,
            temperature=0,
            response_mime_type="application/json",
        )
        contents = _judge_user_payload(question, prediction)
        response = self._generate_with_fallback(
            lambda client: client.models.generate_content(model=self.model, contents=contents, config=config)
        )
        payload = _extract_json_object(response.text or "")
        return _score_from_verdict(question, prediction, payload, judge_spec=self.spec, model=self.model)

    def _generate_with_fallback(self, call: Callable[[Any], Any]) -> Any:
        """Try the active key (with retry); on a quota error roll to the next key."""

        last_exc: Exception | None = None
        while self._index < len(self._clients):
            client = self._clients[self._index]
            try:
                return _with_retry(lambda: call(client))
            except Exception as exc:
                last_exc = exc
                if self._index + 1 < len(self._clients) and _is_quota_error(exc):
                    log.warning(
                        "judge key #%d hit quota/rate limit (%s); falling back to key #%d",
                        self._index + 1,
                        type(exc).__name__,
                        self._index + 2,
                    )
                    self._index += 1
                    continue
                raise
        assert last_exc is not None  # loop entered at least once (>=1 client)
        raise last_exc


def get_judge(spec: str) -> Judge:
    """Return a judge implementation for a config spec."""

    normalized = spec.strip().casefold()
    if normalized in {"stub", ""}:
        return StubJudge("stub")
    if normalized in {"gpt4o-mini", "gpt-4o-mini", "gpt4o-mini-judge", "gpt-4o-mini-judge"}:
        return GPT4oMiniJudge(spec=spec)
    if normalized in {"gemini", "gemini-judge", "gemini-flash", "gemini-flash-judge", "gemini-2.5-flash"}:
        return GeminiJudge(spec=spec)
    raise ValueError(f"unsupported judge spec {spec!r}")
