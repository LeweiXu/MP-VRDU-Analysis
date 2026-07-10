"""Turns a YAML spec into dynamic tasks, including corpus scope."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# A spec declares only what a run varies. There is deliberately no `machine`
# concept: the two-machine reality is the failed-only retry, not a spec field.
ALLOWED_KEYS = {
    "task",
    "representations",
    "corpus",
    "reasoner_spec",
    "reasoner_specs",
    "conditions",
    "parser",
    "quantization",
    "visual_resolution",
    "visual_resolutions",
    "dpi",
    "k_values",
    "judge_spec",
    "run_tag",
}


class SpecError(ValueError):
    """A malformed generation spec."""


@dataclass(frozen=True)
class Spec:
    """A parsed generation spec: which task runs, over what grid and corpus."""

    task: str
    representations: tuple[str, ...] = ("T", "TL", "TLV", "V")
    corpus: Mapping[str, Any] = field(default_factory=lambda: {"sampling": "full"})
    reasoner_spec: str | None = None
    reasoner_specs: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()
    parser: str | None = None
    quantization: str | None = None
    visual_resolution: str | None = None
    visual_resolutions: tuple[str, ...] = ()
    dpi: int | None = None
    k_values: tuple[int, ...] = ()
    judge_spec: str | None = None
    run_tag: str | None = None


def parse_spec(raw: Mapping[str, Any]) -> Spec:
    """Parse a mapping into a `Spec`, rejecting a `machine` field outright."""

    if not isinstance(raw, Mapping):
        raise SpecError(f"spec must be a mapping, got {type(raw).__name__}")
    if "machine" in raw:
        raise SpecError("specs must not declare a machine field; the split is the failed-only retry")
    unknown = set(raw) - ALLOWED_KEYS
    if unknown:
        raise SpecError(f"unknown spec keys: {sorted(unknown)}")
    task = str(raw.get("task") or "").strip()
    if not task:
        raise SpecError("spec must name a task")
    return Spec(
        task=task,
        representations=tuple(raw.get("representations") or ("T", "TL", "TLV", "V")),
        corpus=dict(raw.get("corpus") or {"sampling": "full"}),
        reasoner_spec=raw.get("reasoner_spec"),
        reasoner_specs=tuple(raw.get("reasoner_specs") or ()),
        conditions=tuple(raw.get("conditions") or ()),
        parser=raw.get("parser"),
        quantization=raw.get("quantization"),
        visual_resolution=raw.get("visual_resolution"),
        visual_resolutions=tuple(raw.get("visual_resolutions") or ()),
        dpi=raw.get("dpi"),
        k_values=tuple(raw.get("k_values") or ()),
        judge_spec=raw.get("judge_spec"),
        run_tag=raw.get("run_tag"),
    )


def parse_specs(raw: Mapping[str, Any]) -> list[Spec]:
    """Parse a spec file into one or more runs.

    A flat mapping (with a `task`) is a single run. A mapping with a `runs` list
    is several runs: each run entry is merged over an optional shared `base`
    block, so a file can hold, say, a G1 size sweep and one G1 run per resolution
    plus G3 and G4, each isolated by its own `run_tag`. Splitting a task across
    machines is just which runs each machine's file carries; there is no machine
    field. Every run in a multi-run file must set a unique `run_tag` (its cache
    namespace), so the runs never write over each other.
    """

    if not isinstance(raw, Mapping):
        raise SpecError(f"spec must be a mapping, got {type(raw).__name__}")
    if "runs" not in raw:
        return [parse_spec(raw)]

    unknown_top = set(raw) - {"base", "runs"}
    if unknown_top:
        raise SpecError(f"with `runs`, the only top-level keys are base/runs; unknown: {sorted(unknown_top)}")
    base = raw.get("base") or {}
    if not isinstance(base, Mapping):
        raise SpecError("base must be a mapping")
    runs = raw["runs"]
    if not isinstance(runs, list) or not runs:
        raise SpecError("runs must be a non-empty list")

    specs: list[Spec] = []
    for entry in runs:
        if not isinstance(entry, Mapping):
            raise SpecError(f"each run must be a mapping, got {type(entry).__name__}")
        specs.append(parse_spec({**base, **entry}))

    tags = [spec.run_tag for spec in specs]
    if any(tag is None for tag in tags):
        raise SpecError("every run in a multi-run spec must set run_tag (the cache namespace)")
    if len(set(tags)) != len(tags):
        raise SpecError(f"run_tag must be unique across runs; got {tags}")
    return specs


def load_yaml_spec(path: Path) -> Spec:
    """Read a single-run YAML spec file and parse it into a `Spec`."""

    import yaml

    with Path(path).open() as handle:
        raw = yaml.safe_load(handle)
    return parse_spec(raw)


def load_yaml_specs(path: Path) -> list[Spec]:
    """Read a YAML spec file into its run(s), supporting the `base` + `runs` form."""

    import yaml

    with Path(path).open() as handle:
        raw = yaml.safe_load(handle)
    return parse_specs(raw)


def _sampling(spec: Spec) -> Mapping[str, Any]:
    """The corpus `sampling` block, tolerating `{sampling: {...}}` or a flat block."""

    corpus = spec.corpus or {}
    inner = corpus.get("sampling", corpus)
    return inner if isinstance(inner, Mapping) else {}


def corpus_limit(spec: Spec) -> int | None:
    """A `{sampling: {limit: N}}` cap for the run, or None for the full pool."""

    value = _sampling(spec).get("limit")
    return int(value) if value is not None else None


def config_from_spec(spec: Spec, *, smoke: bool = False):
    """Build an `ExperimentConfig` from a spec's run knobs."""

    from config import DEFAULT_REASONER_SPEC, DEPLOYMENT_RESOLUTION, ExperimentConfig

    sampling = _sampling(spec)
    kwargs: dict[str, Any] = {}
    if "per_bin" in sampling:
        kwargs["per_bin_sample"] = int(sampling["per_bin"])
    if "per_doc_type" in sampling:
        kwargs["per_doc_type_sample"] = int(sampling["per_doc_type"])
    if "seed" in sampling:
        kwargs["sample_seed"] = int(sampling["seed"])
    if spec.conditions:
        kwargs["conditions"] = spec.conditions
    if spec.dpi is not None:
        kwargs["dpi"] = int(spec.dpi)

    return ExperimentConfig(
        smoke=smoke,
        reasoner_spec=spec.reasoner_spec or DEFAULT_REASONER_SPEC,
        reasoner_specs=spec.reasoner_specs,
        representations=spec.representations,
        judge_spec=spec.judge_spec or "stub",
        quantization=spec.quantization,
        visual_resolution=spec.visual_resolution or DEPLOYMENT_RESOLUTION,
        visual_resolutions=spec.visual_resolutions,
        k_values=spec.k_values or (1, 3, 5, 7, 10),
        run_tag=spec.run_tag,
        parser_tool=spec.parser or "paddleocrvl",
        **kwargs,
    )
