"""Config schema + YAML loader (Stage 0).

One run = one config. A config fully specifies representation, retrieval,
generation, judge and dataset slice (Operating Principle 6). Results filenames
encode a hash of the config so any run is reproducible and auditable.

Dataclasses (not pydantic) keep this dependency-free so the config system and
its tests run on the light local env. Validation is explicit: every enum-like
field is checked against an allowed set, with a clear error message.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

# --- allowed values per enum-like field (single source of truth for validation) ---
PARSERS = {"pymupdf4llm", "pymupdf", "mineru", "tesseract", "none"}
CHUNKINGS = {"chunk", "page", "section"}
TEXT_FORMATS = {"markdown", "text", "html"}
RETRIEVAL_METHODS = {
    "none",      # no-retrieval baseline (first-N pages)
    "oracle",    # gold evidence_pages upper bound
    "bm25",
    "tfidf",
    "dense",
    "colpali",
    "colqwen",
    "hybrid",
}
MODALITIES = {"image", "text", "both"}
GENERATORS = {"mock", "local_small_vlm", "kaya_vlm"}
JUDGES = {"rule", "llm"}


class ConfigError(ValueError):
    """Raised when a config fails validation."""


@dataclass
class DataConfig:
    name: str = "mmlongbench-doc"     # dataset key (Stage-1 loader dispatch)
    slice: str = "dev"               # "dev" | "full" | path to a slice manifest
    split: str = "test"
    max_questions: Optional[int] = None  # hard cap, mostly for smoke runs


@dataclass
class RepresentationConfig:
    parser: str = "pymupdf4llm"      # Stage 2 text producer (text retrievers)
    chunking: str = "page"
    dpi: int = 144                   # page-render resolution for images
    text_format: str = "markdown"


@dataclass
class RetrievalConfig:
    method: str = "none"
    top_k: int = 4
    embedding_model: Optional[str] = None   # dense text encoder
    visual_model: Optional[str] = None      # ColPali/ColQwen checkpoint
    no_retrieval_pages: int = 10     # N for the first-N no-retrieval baseline
    hybrid_methods: list[str] = field(default_factory=lambda: ["bm25", "dense"])
    rrf_k: int = 60                  # Reciprocal Rank Fusion constant


@dataclass
class GenerationConfig:
    modality: str = "image"          # what the frozen generator reads
    generator: str = "mock"
    model_id: Optional[str] = None
    mock_mode: str = "gold"          # mock only: "gold" | "wrong" | "echo"
    max_new_tokens: int = 256
    temperature: float = 0.0
    load_in_4bit: bool = False       # quantise real VLMs (fit 12GB; local only)
    device_map: str = "auto"
    # bound vision tokens per page image (Qwen2.5-VL). Keeps many-page inputs
    # within 12GB locally; raise/remove on Kaya. None -> model default.
    max_pixels: Optional[int] = None
    min_pixels: Optional[int] = None


@dataclass
class JudgeConfig:
    type: str = "rule"
    model_id: Optional[str] = None


@dataclass
class RunConfig:
    name: str = "unnamed"
    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    # ----- serialisation / hashing -----
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        """Stable JSON used for hashing. Sorted keys, no whitespace drift."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def hash(self, length: int = 8) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:length]

    # ----- validation -----
    def validate(self) -> "RunConfig":
        _check("representation.parser", self.representation.parser, PARSERS)
        _check("representation.chunking", self.representation.chunking, CHUNKINGS)
        _check("representation.text_format", self.representation.text_format, TEXT_FORMATS)
        _check("retrieval.method", self.retrieval.method, RETRIEVAL_METHODS)
        _check("generation.modality", self.generation.modality, MODALITIES)
        _check("generation.generator", self.generation.generator, GENERATORS)
        _check("judge.type", self.judge.type, JUDGES)
        if self.generation.generator == "mock":
            _check("generation.mock_mode", self.generation.mock_mode,
                   {"gold", "wrong", "echo"})
        if self.retrieval.top_k < 1:
            raise ConfigError("retrieval.top_k must be >= 1")
        if self.retrieval.method == "hybrid":
            fusable = RETRIEVAL_METHODS - {"none", "oracle", "hybrid"}
            if len(self.retrieval.hybrid_methods) < 2:
                raise ConfigError("hybrid needs >= 2 hybrid_methods")
            for m in self.retrieval.hybrid_methods:
                _check("retrieval.hybrid_methods[]", m, fusable)
        if self.representation.dpi < 36:
            raise ConfigError("representation.dpi looks too low (< 36)")
        # Stage-entanglement guard: visual retrievers ignore the text parser, so a
        # text parser paired with a visual retriever is harmless but a *text/both*
        # generation modality over any retriever needs a real parser. Warn-by-raise
        # only on the clearly invalid combo: text modality with parser "none".
        if self.generation.modality in {"text", "both"} and self.representation.parser == "none":
            raise ConfigError(
                "generation.modality is 'text'/'both' but representation.parser is "
                "'none' — text modality needs a parser to produce text."
            )
        return self


def _check(path: str, value: Any, allowed: set[str]) -> None:
    if value not in allowed:
        raise ConfigError(
            f"{path}={value!r} is not one of {sorted(allowed)}"
        )


def _build(cls, data: Optional[dict[str, Any]]):
    """Recursively build a (possibly nested) dataclass from a plain dict.

    Unknown keys raise, so typos in a YAML config fail loudly instead of being
    silently ignored.
    """
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"expected a mapping for {cls.__name__}, got {type(data).__name__}")
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    for fname, f in known.items():
        if fname not in data:
            continue
        ftype = f.type
        # nested dataclass?
        nested_cls = _nested_dataclass(cls, fname)
        if nested_cls is not None:
            kwargs[fname] = _build(nested_cls, data[fname])
        else:
            kwargs[fname] = data[fname]
    return cls(**kwargs)


def _nested_dataclass(cls, fname):
    """Return the dataclass type of a field if it is itself a dataclass."""
    # Resolve via the default_factory instance type, robust to string annotations.
    for f in fields(cls):
        if f.name != fname:
            continue
        if f.default_factory is not MISSING:
            inst = f.default_factory()
            if is_dataclass(inst):
                return type(inst)
    return None


def load_config(path: str | Path) -> RunConfig:
    """Load a YAML file into a validated RunConfig."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = _build(RunConfig, raw)
    return cfg.validate()


def dict_to_config(raw: dict[str, Any]) -> RunConfig:
    """Build + validate a RunConfig from an in-memory dict (used in tests)."""
    return _build(RunConfig, raw).validate()
