"""Parses the flat generation-spec format into runs and their configs.

Each `task_name` is a label; a list-valued axis is the set of values to run over.
Dataset and parser expand to one run each; reasoner_spec x quantization and
visual_resolution become driver-looped lists; representations, k, and prompt modes
are cell dimensions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ALLOWED_KEYS = {
    "task_name",
    "run_tag",
    "dataset",
    "corpus",
    "parser_dpi",
    "parser",
    "retrieval_representation",
    "text_retrievers",
    "vision_retrievers",
    "joints",
    "k_values",
    "joint_k_values",
    "inference_text_retriever",
    "inference_vision_retriever",
    "inference_joint",
    "page_set",
    "reasoner_spec",
    "quantization",
    "visual_resolution",
    "reasoner_representations",
    "prompt_modes",
    "decode_budget",
    "final_answer_delimiter",
    "classifier",
}


class SpecError(ValueError):
    """A malformed generation spec."""


@dataclass(frozen=True)
class Spec:
    """One parsed run: a task_name label plus the full, explicit variable set.

    Already expanded over the dataset / parser cross-product, so a Spec maps 1:1 to
    a driver run. reasoner_specs folds in the quantization grid; visual_resolutions
    is looped by the driver; representations / k / prompt_modes are cell dimensions.
    """

    task_name: str
    run_tag: str | None = None
    dataset: str = "mmlongbench"
    corpus: Mapping[str, Any] = field(default_factory=lambda: {"pool": "answerable", "sampling": "full"})
    parser_dpi: int = 200
    parser: str = "paddleocrvl"
    retrieval_representation: tuple[str, ...] = ()
    text_retrievers: tuple[str, ...] = ()
    vision_retrievers: tuple[str, ...] = ()
    joints: Any = ()
    k_values: tuple[int, ...] = ()
    joint_k_values: tuple[int, ...] = ()
    inference_text_retriever: str | None = None
    inference_vision_retriever: str | None = None
    inference_joint: bool = False
    page_set: Mapping[str, Any] | None = None
    reasoner_specs: tuple[str, ...] = ()
    visual_resolutions: tuple[str, ...] = ()
    reasoner_representations: tuple[str, ...] = ("T", "TL", "TLV", "V")
    prompt_modes: tuple[str, ...] = ("none",)
    decode_budget: Mapping[str, int] | None = None
    final_answer_delimiter: str | None = None
    classifier: str | None = None


def _as_list(value: Any) -> list[Any]:
    """A scalar becomes a one-element list; a list/tuple passes through."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _clean_none(value: Any) -> str | None:
    """Map an unset / literal 'none' string to None."""
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in ("", "none") else text


def _is_bf16(value: Any) -> bool:
    """A quantization value meaning the unquantized baseline (no suffix)."""
    return not value or (isinstance(value, str) and value.strip().lower() in ("bf16", "none", ""))


def _parse_decode_budget(value: Any, *, run_tag: str) -> Mapping[str, int] | None:
    """Validate a per-prompt-mode decode budget block.

    Must be a mapping with a `default` entry; every other key must be a known
    prompt mode; values positive ints. Budgets are run_tag-scoped (never part of
    the cell key), so getting them right at parse time is the safety line.
    """

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SpecError(f"{run_tag}: decode_budget must be a mapping, got {type(value).__name__}")
    from config import PROMPT_MODES

    budget: dict[str, int] = {}
    for key, entry in value.items():
        name = str(key)
        if name != "default" and name not in PROMPT_MODES:
            raise SpecError(f"{run_tag}: decode_budget names unknown prompt mode {name!r}")
        try:
            tokens = int(entry)
        except (TypeError, ValueError):
            raise SpecError(f"{run_tag}: decode_budget[{name!r}] must be an int, got {entry!r}") from None
        if tokens <= 0:
            raise SpecError(f"{run_tag}: decode_budget[{name!r}] must be positive, got {tokens}")
        budget[name] = tokens
    if "default" not in budget:
        raise SpecError(f"{run_tag}: decode_budget must contain a 'default' entry")
    return budget


_CORPUS_KEYS = {"scan", "pool", "sampling", "hop"}
_PAGE_SET_KEYS = {"ranking_source", "gold", "distractor", "on_insufficient_gold",
                  "on_insufficient_distractors", "on_no_gold"}


def _parse_page_set(value: Any, *, run_tag: str, corpus: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Validate a `page_set` block into its normalized mapping (or None).

    Every (ranking_source, distractor count) combination is materialised as a
    `PageSetRule` here so its vocabulary/consistency checks run at parse time,
    not at cell time. Gold rules that remove or isolate a page require
    `corpus.hop: multi` (at one gold page, top and bottom coincide).
    """

    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "none")):
        return None
    if not isinstance(value, Mapping):
        raise SpecError(f"{run_tag}: page_set must be a mapping or 'none', got {type(value).__name__}")
    unknown = set(value) - _PAGE_SET_KEYS
    if unknown:
        raise SpecError(f"{run_tag}: unknown page_set keys: {sorted(unknown)}")

    from pipeline.page_rules import PageSetRule

    rankers = [str(r) for r in _as_list(value.get("ranking_source"))]
    if not rankers:
        raise SpecError(f"{run_tag}: page_set must name at least one ranking_source")
    gold = value.get("gold") or {"mode": "all"}
    if not isinstance(gold, Mapping) or set(gold) - {"mode", "count"}:
        raise SpecError(f"{run_tag}: page_set.gold must be a mapping with mode/count, got {gold!r}")
    gold_mode = str(gold.get("mode", "all"))
    gold_count = int(gold.get("count", 0)) if gold_mode != "all" else 0
    distractor = value.get("distractor") or {"count": 0}
    if not isinstance(distractor, Mapping) or set(distractor) - {"count"}:
        raise SpecError(f"{run_tag}: page_set.distractor must be a mapping with count, got {distractor!r}")
    d_counts = [int(d) for d in _as_list(distractor.get("count"))] or [0]

    normalized = {
        "ranking_source": tuple(rankers),
        "gold": {"mode": gold_mode, "count": gold_count},
        "distractor": {"count": tuple(d_counts)},
        "on_insufficient_gold": str(value.get("on_insufficient_gold", "exclude")),
        "on_insufficient_distractors": str(value.get("on_insufficient_distractors", "pad_available")),
        "on_no_gold": str(value.get("on_no_gold", "exclude")),
    }
    try:
        for ranker in rankers:
            for d in d_counts:
                PageSetRule(
                    ranking_source=ranker, gold_mode=gold_mode, gold_count=gold_count,
                    distractor_count=d,
                    on_insufficient_gold=normalized["on_insufficient_gold"],
                    on_insufficient_distractors=normalized["on_insufficient_distractors"],
                    on_no_gold=normalized["on_no_gold"],
                )
    except ValueError as exc:
        raise SpecError(f"{run_tag}: invalid page_set: {exc}") from None
    hop = str(corpus.get("hop", "any"))
    if gold_mode != "all" and hop != "multi" and not (hop.isdigit() and int(hop) >= 2):
        raise SpecError(
            f"{run_tag}: page_set gold mode {gold_mode!r} requires corpus.hop: multi "
            f"(or an exact count >= 2); at one gold page, top and bottom coincide"
        )
    return normalized


def _expand_run(raw: Mapping[str, Any]) -> list[Spec]:
    """Expand one flat run mapping into its Spec(s) (dataset x parser cross-product)."""

    from config import DEFAULT_REASONER_SPEC

    if not isinstance(raw, Mapping):
        raise SpecError(f"run must be a mapping, got {type(raw).__name__}")
    unknown = set(raw) - ALLOWED_KEYS
    if unknown:
        raise SpecError(f"unknown spec keys: {sorted(unknown)}")
    task_name = str(raw.get("task_name") or "").strip()
    if not task_name:
        raise SpecError("spec must name a task_name")
    run_tag = raw.get("run_tag") or task_name

    datasets = _as_list(raw.get("dataset")) or ["mmlongbench"]
    parsers = _as_list(raw.get("parser")) or ["paddleocrvl"]
    quant_list = _as_list(raw.get("quantization")) or ["bf16"]
    reasoner_list = _as_list(raw.get("reasoner_spec")) or [DEFAULT_REASONER_SPEC]
    reasoner_specs = tuple(
        r if _is_bf16(q) else f"{r}-{q}" for r in reasoner_list for q in quant_list
    )
    visual_resolutions = tuple(_as_list(raw.get("visual_resolution")) or ["med"])
    from config import DEFAULT_REPRESENTATIONS

    reps = tuple(raw.get("reasoner_representations") or DEFAULT_REPRESENTATIONS)

    retrieval_representation = tuple(raw.get("retrieval_representation") or ())
    text_retrievers = tuple(raw.get("text_retrievers") or ())
    vision_retrievers = tuple(raw.get("vision_retrievers") or ())
    joints_raw = raw.get("joints", "matched")
    joints: Any = () if joints_raw in ([], None) else joints_raw
    k_values = tuple(int(k) for k in (raw.get("k_values") or ()))
    joint_k_values = tuple(int(k) for k in (raw.get("joint_k_values") or ()))
    inf_text = _clean_none(raw.get("inference_text_retriever"))
    inf_vision = _clean_none(raw.get("inference_vision_retriever"))

    # Enforcement: a run whose retrieval benchmark also FEEDS the reasoner
    # (inference arms set) must include the fixed inference arms so their
    # rankings are the ones the reasoner reuses, and any inference pick must be
    # a benchmarked method. A benchmark that only supplies page_set rankings
    # (inference arms none) may list any methods.
    if (text_retrievers or vision_retrievers) and (inf_text or inf_vision):
        if "bge-m3" not in text_retrievers:
            raise SpecError(f"{run_tag}: a retrieval benchmark must list bge-m3 in text_retrievers")
        if "colqwen2.5" not in vision_retrievers:
            raise SpecError(f"{run_tag}: a retrieval benchmark must list colqwen2.5 in vision_retrievers")
        if inf_text and inf_text not in text_retrievers:
            raise SpecError(f"{run_tag}: inference text_retriever {inf_text!r} not in {text_retrievers}")
        if inf_vision and inf_vision not in vision_retrievers:
            raise SpecError(f"{run_tag}: inference vision_retriever {inf_vision!r} not in {vision_retrievers}")

    corpus = dict(raw.get("corpus") or {"pool": "answerable", "sampling": "full"})
    unknown_corpus = set(corpus) - _CORPUS_KEYS
    if unknown_corpus:
        raise SpecError(f"{run_tag}: unknown corpus keys: {sorted(unknown_corpus)}")
    page_set = _parse_page_set(raw.get("page_set"), run_tag=str(run_tag), corpus=corpus)
    parser_dpi = int(raw.get("parser_dpi", 200))
    prompt_modes = tuple(raw.get("prompt_modes") or ("none",))
    decode_budget = _parse_decode_budget(raw.get("decode_budget"), run_tag=str(run_tag))
    final_answer_delimiter = _clean_none(raw.get("final_answer_delimiter"))
    classifier = _clean_none(raw.get("classifier"))

    combos = [(ds, ps) for ds in datasets for ps in parsers]
    specs: list[Spec] = []
    for ds, ps in combos:
        tag = run_tag
        if len(datasets) > 1:
            tag = f"{tag}-{ds}"
        if len(parsers) > 1:
            tag = f"{tag}-{ps}"
        specs.append(Spec(
            task_name=task_name,
            run_tag=tag,
            dataset=str(ds),
            corpus=corpus,
            parser_dpi=parser_dpi,
            parser=str(ps),
            retrieval_representation=retrieval_representation,
            text_retrievers=text_retrievers,
            vision_retrievers=vision_retrievers,
            joints=joints,
            k_values=k_values,
            joint_k_values=joint_k_values,
            inference_text_retriever=inf_text,
            inference_vision_retriever=inf_vision,
            inference_joint=bool(raw.get("inference_joint", False)),
            page_set=page_set,
            reasoner_specs=reasoner_specs,
            visual_resolutions=visual_resolutions,
            reasoner_representations=reps,
            prompt_modes=prompt_modes,
            decode_budget=decode_budget,
            final_answer_delimiter=final_answer_delimiter,
            classifier=classifier,
        ))
    return specs


def parse_specs(raw: Mapping[str, Any]) -> list[Spec]:
    """Parse a spec file into its run(s): a `{runs: [...]}` list or a single flat run."""

    if not isinstance(raw, Mapping):
        raise SpecError(f"spec must be a mapping, got {type(raw).__name__}")
    if "runs" in raw:
        runs = raw.get("runs")
        if not isinstance(runs, Sequence) or not runs:
            raise SpecError("runs must be a non-empty list")
        specs: list[Spec] = []
        seen: set[str] = set()
        for entry in runs:
            for spec in _expand_run(entry):
                if spec.run_tag in seen:
                    raise SpecError(f"run_tag must be unique across runs; repeated {spec.run_tag!r}")
                seen.add(spec.run_tag)
                specs.append(spec)
        return specs
    return _expand_run(raw)


def parse_spec(raw: Mapping[str, Any]) -> Spec:
    """Parse a single flat run mapping into one Spec (first dataset/parser combo)."""

    specs = _expand_run(raw)
    return specs[0]


def load_yaml_spec(path: Path) -> Spec:
    """Read a single-run YAML spec file into a `Spec`."""

    import yaml

    with Path(path).open() as handle:
        raw = yaml.safe_load(handle)
    return parse_spec(raw)


def load_yaml_specs(path: Path) -> list[Spec]:
    """Read a YAML spec file into its run(s)."""

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
    """Build an `ExperimentConfig` from a parsed flat Spec."""

    from config import DEFAULT_REASONER_SPEC, DEPLOYMENT_RESOLUTION, ExperimentConfig

    sampling = _sampling(spec)
    kwargs: dict[str, Any] = {}
    if "per_bin" in sampling:
        kwargs["per_bin_sample"] = int(sampling["per_bin"])
    if "per_doc_type" in sampling:
        kwargs["per_doc_type_sample"] = int(sampling["per_doc_type"])
    if "seed" in sampling:
        kwargs["sample_seed"] = int(sampling["seed"])

    reasoner_spec = spec.reasoner_specs[0] if spec.reasoner_specs else DEFAULT_REASONER_SPEC
    return ExperimentConfig(
        smoke=smoke,
        pool=str(spec.corpus.get("pool", "answerable")),
        scan_filter=str(spec.corpus.get("scan", "any")),
        hop_filter=str(spec.corpus.get("hop", "any")),
        page_set=spec.page_set,
        sampling=spec.corpus.get("sampling", "full"),
        reasoner_spec=reasoner_spec,
        reasoner_specs=spec.reasoner_specs,
        representations=spec.reasoner_representations,
        retrieval_representation=spec.retrieval_representation,
        visual_resolution=spec.visual_resolutions[0] if spec.visual_resolutions else DEPLOYMENT_RESOLUTION,
        visual_resolutions=spec.visual_resolutions,
        k_values=spec.k_values or (1,),
        joint_k_values=spec.joint_k_values or (1, 3, 5),
        text_retrievers=spec.text_retrievers,
        vision_retrievers=spec.vision_retrievers,
        joints=spec.joints if spec.joints is not None else (),
        inference_text_retriever=spec.inference_text_retriever or "none",
        inference_vision_retriever=spec.inference_vision_retriever or "none",
        inference_joint=bool(spec.inference_joint),
        inference_representations=spec.reasoner_representations,
        prompt_modes=spec.prompt_modes,
        decode_budget=spec.decode_budget,
        final_answer_delimiter=spec.final_answer_delimiter,
        run_tag=spec.run_tag,
        parser_tool=spec.parser,
        dpi=spec.parser_dpi,
        classifier_spec=spec.classifier,
        dataset=spec.dataset,
        **kwargs,
    )
