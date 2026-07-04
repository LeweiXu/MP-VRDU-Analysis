"""Resolve experiment model specs to reasoner backend instances.

Purpose:
    Implements the model-family swap point named in the plan. `get_reasoner()`
    parses a spec string, selects a backend, and returns a `Reasoner`; pipeline
    code never imports concrete local/API backend classes directly.

Pipeline role:
    The orchestrator asks this registry for the configured reasoner. Adding
    Qwen3-VL sizes, InternVL, or hosted comparison models should be a registry
    change behind the frozen `Reasoner` and `ModelInput` contracts.

Spec grammar: ``<family>-<size>-<backend>`` (e.g. ``qwen3vl-8b-local``,
``gpt4o-api``), or the literal ``stub``. Stage 3 resolves every spec to the
`StubReasoner`; Stage 6 wires the ``local`` backend to `LocalVLMBackend`
(Qwen3-VL etc.) and the ``api`` backend to `APIBackend` (OpenAI / Gemini /
Anthropic-style HTTP). Adding a family is a new registry entry; no pipeline code
changes.

Arguments:
    None. This module is import-only; callers pass a spec string to
    `ModelSpec.parse()` or `get_reasoner()`.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.reasoner import Reasoner, StubReasoner


@dataclass(frozen=True)
class ModelSpec:
    """A parsed model spec: family, size, and backend, plus the raw name."""

    name: str
    family: str
    size: str
    backend: str

    @classmethod
    def parse(cls, spec: str) -> "ModelSpec":
        """Parse a ``family-size-backend`` (or ``stub``) spec string."""

        raw = spec.strip()
        if raw == "stub":
            return cls(name="stub", family="stub", size="", backend="stub")
        parts = raw.split("-")
        if len(parts) < 2:
            raise ValueError(
                f"model spec {spec!r} must be 'family-size-backend' or 'stub'"
            )
        backend = parts[-1]
        if backend in ("local", "api"):
            family = parts[0]
            size = "-".join(parts[1:-1])
        else:
            # No explicit backend suffix; treat the whole thing as family-size.
            family = parts[0]
            size = "-".join(parts[1:])
            backend = "local"
        return cls(name=raw, family=family, size=size, backend=backend)


def get_reasoner(spec: str) -> Reasoner:
    """Return a `Reasoner` for a model spec (the family swap point)."""

    parsed = ModelSpec.parse(spec)
    if parsed.backend == "stub":
        return StubReasoner(spec="stub")
    # Stage 6 dispatches parsed.backend to LocalVLMBackend / APIBackend here.
    # Until then every spec resolves to the stub so the pipeline is runnable.
    return StubReasoner(spec=parsed.name)
