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
    "classifier",
    "dataset",
    "run_tag",
}


class SpecError(ValueError):
    """A malformed generation spec."""


@dataclass(frozen=True)
class Spec:
    """A parsed generation spec: which task runs, over what grid and corpus.

    A flat spec maps 1:1 to a run. The nested `base` + `sweeps` (and G2
    `retrieval` / `inference`) form is expanded into several of these at parse
    time, one per sweep, each with its own `run_tag`.
    """

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
    classifier: str | None = None
    dataset: str | None = None
    run_tag: str | None = None
    # G3 abstention-prompt modes (run as cells within the one G3 run).
    prompt_modes: tuple[str, ...] = ()
    # G2 retrieval-benchmark method sets + inference-stage picks.
    text_retrievers: tuple[str, ...] = ()
    vision_retrievers: tuple[str, ...] = ()
    joints: Any = None
    joint_k_values: tuple[int, ...] = ()
    inference_text_retriever: str | None = None
    inference_vision_retriever: str | None = None
    inference_joint: bool | None = None
    inference_representations: tuple[str, ...] = ()


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
        classifier=raw.get("classifier"),
        dataset=raw.get("dataset"),
        run_tag=raw.get("run_tag"),
    )


# A run entry is "nested" (needs the expander) if it carries any of these blocks.
_NESTED_KEYS = ("base", "sweeps", "retrieval", "inference")
# Sweep axes that live in the cache key (reasoner spec incl. quant suffix, and
# visual_resolution), so a whole sweep runs under ONE run_tag as a list the driver
# loops. Axes NOT in the key (parser, dataset) get one run_tag per value instead.
_DEFAULT_REPS = ("T", "TL", "TLV", "V")


def _is_bf16(value: Any) -> bool:
    """A quantization value that means the unquantized baseline (no suffix)."""

    return not value or (isinstance(value, str) and value.strip().lower() in ("bf16", "none", ""))


def parse_specs(raw: Mapping[str, Any]) -> list[Spec]:
    """Parse a spec file into one or more runs.

    A flat mapping (with a `task`) is a single run. A mapping with a `runs` list is
    several runs, each merged over an optional shared `base` block. A run entry that
    carries a nested `base` / `sweeps` block (or G2's `retrieval` / `inference`) is
    expanded here into one flat `Spec` per sweep, each with its own `run_tag`, so the
    driver still runs one pass per spec. Every emitted run needs a unique `run_tag`.
    """

    if not isinstance(raw, Mapping):
        raise SpecError(f"spec must be a mapping, got {type(raw).__name__}")
    if "runs" not in raw:
        return [parse_spec(raw)]

    unknown_top = set(raw) - {"base", "runs"}
    if unknown_top:
        raise SpecError(f"with `runs`, the only top-level keys are base/runs; unknown: {sorted(unknown_top)}")
    file_base = raw.get("base") or {}
    if not isinstance(file_base, Mapping):
        raise SpecError("base must be a mapping")
    runs = raw["runs"]
    if not isinstance(runs, list) or not runs:
        raise SpecError("runs must be a non-empty list")

    specs: list[Spec] = []
    for entry in runs:
        if not isinstance(entry, Mapping):
            raise SpecError(f"each run must be a mapping, got {type(entry).__name__}")
        if any(key in entry for key in _NESTED_KEYS):
            specs.extend(_expand_run(file_base, entry))
        else:
            specs.append(parse_spec({**file_base, **entry}))

    tags = [spec.run_tag for spec in specs]
    if any(tag is None for tag in tags):
        raise SpecError("every run in a multi-run spec must set run_tag (the cache namespace)")
    if len(set(tags)) != len(tags):
        raise SpecError(f"run_tag must be unique across runs; got {tags}")
    return specs


def _expand_run(file_base: Mapping[str, Any], entry: Mapping[str, Any]) -> list[Spec]:
    """Expand one nested run entry into its flat per-sweep `Spec`s."""

    task = str(entry.get("task") or "").strip()
    run_tag = entry.get("run_tag")
    if not task:
        raise SpecError("nested run must name a task")
    if not run_tag:
        raise SpecError(f"nested run {task!r} must set run_tag")
    if "retrieval" in entry or "inference" in entry:
        return [_g2_spec(file_base, entry, task, run_tag)]
    return _expand_reasoner_run(file_base, entry, task, run_tag)


def _baseline_spec(file_base, task_base, task, run_tag, *, prompt_modes) -> Spec:
    """The `base` run: file-level base overlaid by the task base (task wins).

    Unknown task-base keys (e.g. G3's informational `retriever` / `similarity_k` /
    singular `representation`) are ignored; only mapped axes take effect.
    """

    merged = {**file_base, **task_base}
    quant = None if _is_bf16(merged.get("quantization")) else merged.get("quantization")
    return Spec(
        task=task,
        run_tag=run_tag,
        representations=tuple(merged.get("representations") or _DEFAULT_REPS),
        corpus=dict(merged.get("corpus") or {"sampling": "full"}),
        reasoner_spec=merged.get("reasoner_spec"),
        parser=merged.get("parser"),
        quantization=quant,
        visual_resolution=merged.get("visual_resolution"),
        dpi=merged.get("dpi"),
        k_values=tuple(merged.get("k_values") or ()),
        judge_spec=merged.get("judge_spec"),
        classifier=merged.get("classifier"),
        dataset=merged.get("dataset"),
        prompt_modes=tuple(prompt_modes or ()),
    )


def _expand_reasoner_run(file_base, entry, task, run_tag) -> list[Spec]:
    """G1/G3-style: a base run plus one Spec (or run_tag) per enabled sweep."""

    from dataclasses import replace

    task_base = entry.get("base") or {}
    if not isinstance(task_base, Mapping):
        raise SpecError(f"{run_tag}: base must be a mapping")
    sweeps = entry.get("sweeps") or {}
    if not isinstance(sweeps, Mapping):
        raise SpecError(f"{run_tag}: sweeps must be a mapping")

    # A `prompt_mode` sweep configures the single G3 run (its modes run as cells),
    # so fold it into the base rather than emitting a separate run_tag.
    prompt_modes = None
    for sweep in sweeps.values():
        if isinstance(sweep, Mapping) and "prompt_mode" in sweep:
            prompt_modes = tuple(sweep["prompt_mode"])
    baseline = _baseline_spec(file_base, task_base, task, run_tag, prompt_modes=prompt_modes)

    specs = [baseline]
    for name, sweep in sweeps.items():
        if not sweep or (isinstance(sweep, Mapping) and sweep.get("enabled") is False):
            continue
        if not isinstance(sweep, Mapping):
            raise SpecError(f"{run_tag}: sweep {name!r} must be a mapping")
        axes = [k for k in sweep if k not in ("representations", "enabled")]
        if len(axes) != 1:
            raise SpecError(f"{run_tag}: sweep {name!r} must vary exactly one axis, got {axes}")
        axis = axes[0]
        if axis == "prompt_mode":
            continue  # already folded into the base run
        values = list(sweep[axis])
        if not values:
            continue
        reps = {"representations": tuple(sweep["representations"])} if sweep.get("representations") else {}
        specs.extend(_emit_sweep(baseline, replace, run_tag, name, axis, values, reps))
    return specs


def _emit_sweep(baseline, replace, run_tag, name, axis, values, reps) -> list[Spec]:
    """One sweep -> its Spec(s), per the run_tag strategy (see module docs)."""

    if axis == "reasoner_spec":  # size and family sweeps both vary this axis
        if len(values) == 1 and values[0] == baseline.reasoner_spec:
            return []
        return [replace(baseline, run_tag=f"{run_tag}-{name}", reasoner_specs=tuple(values), **reps)]
    if axis == "quantization":
        if len(values) == 1 and _is_bf16(values[0]) and baseline.quantization is None:
            return []
        from config import DEFAULT_REASONER_SPEC
        base_reasoner = baseline.reasoner_spec or DEFAULT_REASONER_SPEC
        rspecs = tuple(base_reasoner if _is_bf16(v) else f"{base_reasoner}-{v}" for v in values)
        return [replace(baseline, run_tag=f"{run_tag}-{name}", reasoner_specs=rspecs, quantization=None, **reps)]
    if axis == "visual_resolution":
        if len(values) == 1 and values[0] == baseline.visual_resolution:
            return []
        return [replace(baseline, run_tag=f"{run_tag}-{name}", visual_resolutions=tuple(values), **reps)]
    if axis in ("parser", "dataset"):
        # Not in the cache key: each value needs its own run_tag namespace.
        return [replace(baseline, run_tag=f"{run_tag}-{name}-{v}", **{axis: v}, **reps) for v in values]
    raise SpecError(f"{run_tag}: sweep {name!r} varies unknown axis {axis!r}")


def _g2_spec(file_base, entry, task, run_tag) -> Spec:
    """G2: one Spec carrying the retrieval-benchmark method sets and the inference
    picks. Inference retrievers must be a subset of the benchmark lists (they reuse
    the stage-1 cached rankings)."""

    retrieval = entry.get("retrieval") or {}
    inference = entry.get("inference") or {}
    if not isinstance(retrieval, Mapping) or not isinstance(inference, Mapping):
        raise SpecError(f"{run_tag}: retrieval/inference must be mappings")

    text_list = tuple(retrieval.get("text_retrievers") or ())
    vision_list = tuple(retrieval.get("vision_retrievers") or ())
    inf_text = inference.get("text_retriever")
    inf_vision = inference.get("vision_retriever")
    if inf_text and text_list and inf_text not in text_list:
        raise SpecError(f"{run_tag}: inference text_retriever {inf_text!r} not in retrieval set {text_list}")
    if inf_vision and vision_list and inf_vision not in vision_list:
        raise SpecError(f"{run_tag}: inference vision_retriever {inf_vision!r} not in retrieval set {vision_list}")

    k_values = tuple(inference.get("k_values") or retrieval.get("k_values") or ())
    joint_ks = tuple(inference.get("joint_k_values") or retrieval.get("joint_k_values") or ())
    return Spec(
        task=task,
        run_tag=run_tag,
        corpus=dict(file_base.get("corpus") or {"sampling": "full"}),
        dataset=file_base.get("dataset"),
        parser=file_base.get("parser"),
        judge_spec=file_base.get("judge_spec"),
        reasoner_spec=inference.get("reasoner_spec"),
        visual_resolution=inference.get("visual_resolution"),
        k_values=k_values,
        text_retrievers=text_list,
        vision_retrievers=vision_list,
        joints=retrieval.get("joints", "matched"),
        joint_k_values=joint_ks,
        inference_text_retriever=inf_text,
        inference_vision_retriever=inf_vision,
        inference_joint=bool(inference.get("joint", True)),
        inference_representations=tuple(inference.get("representations") or ("TLV", "V")),
    )


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

    # Only override an ExperimentConfig default when the spec actually set the field,
    # so a plain run keeps the class defaults for every axis it does not name.
    if spec.dataset:
        kwargs["dataset"] = spec.dataset
    if spec.prompt_modes:
        kwargs["prompt_modes"] = spec.prompt_modes
    if spec.text_retrievers:
        kwargs["text_retrievers"] = spec.text_retrievers
    if spec.vision_retrievers:
        kwargs["vision_retrievers"] = spec.vision_retrievers
    if spec.joints is not None:
        kwargs["joints"] = spec.joints
    if spec.joint_k_values:
        kwargs["joint_k_values"] = spec.joint_k_values
    if spec.inference_text_retriever:
        kwargs["inference_text_retriever"] = spec.inference_text_retriever
    if spec.inference_vision_retriever:
        kwargs["inference_vision_retriever"] = spec.inference_vision_retriever
    if spec.inference_joint is not None:
        kwargs["inference_joint"] = spec.inference_joint
    if spec.inference_representations:
        kwargs["inference_representations"] = spec.inference_representations

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
        classifier_spec=spec.classifier,
        **kwargs,
    )
