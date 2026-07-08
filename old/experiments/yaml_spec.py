"""Load YAML generation specs into dynamic experiment tasks.

Purpose:
    Provides the YAML-first data-collection interface. A spec file describes one
    or more generation runs as explicit cell grids over questions, conditions,
    representations, and model specs. The resulting objects satisfy the existing
    `GenerationTask` contract so the current driver/orchestrator/cache machinery
    remains the execution engine.

Pipeline role:
    `cli.generate --spec <file.yaml>` loads this module, builds an
    `ExperimentConfig`, resolves the shared corpus, and runs each dynamic task.
    The judge/build phases can later consume the manifest and cache artifacts
    without needing to re-specify the generation flags.

Arguments:
    None. Import-only. Public entry point: `load_yaml_experiment(path)`.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from config import ExperimentConfig
from data.binning import doc_type_bin
from experiments.base import Cell, GenerationTask, Retrievers
from experiments.corpus import sample_table4_replication
from experiments.paths import experiment_paths
from experiments.side_artifacts import write_classifier_eval, write_retrieval_eval
from pipeline.conditioner import BuriedOracle, FullDoc, InputConditioner, OracleConditioner, RetrievedTopK
from schema import Modality, Question
from tools.text import ANNOTATION_SHEET


class SpecError(ValueError):
    """Raised when a YAML experiment spec is invalid."""


def _as_list(value: Any) -> list[Any]:
    """Return `value` as a list, treating scalars as a one-item list."""

    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _as_tuple(value: Any) -> tuple:
    return tuple(_as_list(value))


def _read_scan_labels(path: Path = ANNOTATION_SHEET) -> dict[str, str]:
    """Return doc_id -> scan label from the annotation sheet."""

    if not path.exists():
        return {}
    labels: dict[str, str] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            doc_id = (row.get("doc_id") or "").strip()
            label = (row.get("scan_label") or "").strip() or (row.get("auto_scan") or "").strip()
            if doc_id and label in {"scanned", "digital"}:
                labels[doc_id] = label
    return labels


def _validate_name(name: str, *, field_name: str = "name") -> str:
    """Validate a cache-safe YAML identifier."""

    if not name or not all(ch.isalnum() or ch in "-_" for ch in name):
        raise SpecError(f"{field_name} must be non-empty and contain only letters, numbers, dash, underscore: {name!r}")
    return name


def _config_from_spec(raw: Mapping[str, Any]) -> ExperimentConfig:
    """Build an ExperimentConfig from top-level YAML fields."""

    config_values = dict(raw.get("config") or {})
    mode = raw.get("mode", "smoke")
    if mode not in {"smoke", "full"}:
        raise SpecError("mode must be 'smoke' or 'full'")
    config_values["smoke"] = mode == "smoke"
    if "dataset" in raw:
        config_values["dataset"] = raw["dataset"]
    if "run_tag" in raw:
        config_values["run_tag"] = _validate_name(str(raw["run_tag"]), field_name="run_tag")

    tuple_fields = {"k_values", "representations", "bins", "scaling_specs", "conditions", "burying_levels"}
    for field_name in tuple_fields:
        if field_name in config_values:
            config_values[field_name] = tuple(config_values[field_name])
    return ExperimentConfig(**config_values)


@dataclass(frozen=True)
class ConditionSpec:
    """One YAML condition entry after scalar/list expansion."""

    type: str
    retriever: str | None = None
    k: int | None = None
    n_distractors: int | None = None
    name_template: str | None = None

    def build(self, retrievers: Retrievers) -> InputConditioner:
        if self.type == "oracle":
            return OracleConditioner()
        if self.type == "full":
            return FullDoc()
        if self.type == "buried":
            return BuriedOracle(int(self.n_distractors or 0))
        if self.type == "retrieved":
            if self.retriever not in {"text", "vision"}:
                raise SpecError("retrieved condition needs retriever: text|vision")
            k = int(self.k or 1)
            retriever = retrievers.text if self.retriever == "text" else retrievers.vision
            name = (self.name_template or "retrieved_{retriever}_k{k}").format(
                retriever=self.retriever,
                k=k,
            )
            return RetrievedTopK(retriever, k, name=name)
        raise SpecError(f"unknown condition type {self.type!r}")


def _condition_specs(raw_conditions: Sequence[Any]) -> tuple[ConditionSpec, ...]:
    """Expand YAML condition entries into concrete condition specs."""

    specs: list[ConditionSpec] = []
    for item in raw_conditions:
        if isinstance(item, str):
            item = {"type": item}
        if not isinstance(item, Mapping):
            raise SpecError(f"condition entries must be strings or mappings, got {item!r}")
        typ = str(item.get("type") or "")
        if typ not in {"oracle", "full", "buried", "retrieved"}:
            raise SpecError(f"unknown condition type {typ!r}")
        if typ == "retrieved":
            for retriever in _as_list(item.get("retriever", "text")):
                for k in _as_list(item.get("k", 1)):
                    specs.append(
                        ConditionSpec(
                            type=typ,
                            retriever=str(retriever),
                            k=int(k),
                            name_template=item.get("name"),
                        )
                    )
        elif typ == "buried":
            for n in _as_list(item.get("n_distractors", item.get("n", 10))):
                specs.append(ConditionSpec(type=typ, n_distractors=int(n)))
        else:
            specs.append(ConditionSpec(type=typ))
    return tuple(specs)


def _validate_representations(values: Sequence[Any]) -> tuple[Modality, ...]:
    """Validate YAML representation names against the dynamic representation set."""

    from pipeline.representation import VALID_REPRESENTATIONS

    reps = tuple(str(value) for value in values)
    if not reps:
        raise SpecError("cells.representations must contain at least one representation")
    invalid = [rep for rep in reps if rep not in VALID_REPRESENTATIONS]
    if invalid:
        raise SpecError(f"invalid representation(s) {invalid}; choose from {sorted(VALID_REPRESENTATIONS)}")
    return reps


@dataclass(frozen=True)
class QuestionSelector:
    """Question filtering rules from one YAML run."""

    raw: Mapping[str, Any] = field(default_factory=dict)

    def select(self, config: ExperimentConfig, questions: Sequence[Question]) -> list[Question]:
        selected = list(questions)
        raw = self.raw
        if raw.get("preset") == "table4_replication":
            all_bins = tuple(b for b in config.bins if b != "visual_heavy")
            from data.loader import load_mmlongbench

            selected = sample_table4_replication(
                list(load_mmlongbench(data_dir=config.paths.data_dir)),
                config.per_bin_sample or 100,
                bins=all_bins,
                seed=config.sample_seed,
                reuse_bins=(),
            )

        if raw.get("question_ids"):
            allowed = {str(x) for x in _as_list(raw["question_ids"])}
            selected = [q for q in selected if q.id in allowed]
        if raw.get("doc_ids"):
            allowed = {str(x) for x in _as_list(raw["doc_ids"])}
            selected = [q for q in selected if q.doc_id in allowed]
        if raw.get("doc_types"):
            allowed = {str(x) for x in _as_list(raw["doc_types"])}
            selected = [q for q in selected if q.doc_type in allowed]
        if raw.get("bins"):
            allowed = {str(x) for x in _as_list(raw["bins"])}
            selected = [q for q in selected if doc_type_bin(q.doc_type) in allowed]
        if raw.get("scan_labels"):
            labels = _read_scan_labels()
            allowed = {str(x) for x in _as_list(raw["scan_labels"])}
            selected = [q for q in selected if labels.get(q.doc_id) in allowed]
        if raw.get("max_per_doc") is not None:
            max_per_doc = int(raw["max_per_doc"])
            counts: dict[str, int] = defaultdict(int)
            limited: list[Question] = []
            for question in selected:
                if counts[question.doc_id] >= max_per_doc:
                    continue
                counts[question.doc_id] += 1
                limited.append(question)
            selected = limited
        if raw.get("limit") is not None:
            selected = selected[: max(1, int(raw["limit"]))]
        return selected


class YamlGenerationTask(GenerationTask):
    """A dynamic generation task created from one YAML `runs` entry."""

    def __init__(
        self,
        *,
        name: str,
        models: tuple[str, ...],
        selector: QuestionSelector,
        conditions: tuple[ConditionSpec, ...],
        representations: tuple[Modality, ...],
        retrieval_eval: bool = False,
        classifier_eval: bool = False,
    ) -> None:
        self.name = name
        self.models = models
        self.selector = selector
        self.conditions = conditions
        self.representations = representations
        self.retrieval_eval = retrieval_eval
        self.classifier_eval = classifier_eval

    @property
    def side_artifact(self) -> str | None:  # type: ignore[override]
        if self.retrieval_eval:
            return "retrieval.jsonl"
        if self.classifier_eval:
            return "classifier.jsonl"
        return None

    def model_specs(self, config: ExperimentConfig) -> tuple[str, ...]:
        return self.models

    def resolve_questions(self, config: ExperimentConfig, questions: Sequence[Question]) -> Sequence[Question]:
        return self.selector.select(config, questions)

    def generation_cells(self, config: ExperimentConfig, questions: Sequence[Question], *, retrievers: Retrievers) -> list[Cell]:
        cells: list[Cell] = []
        conditioners = tuple(condition.build(retrievers) for condition in self.conditions)
        for question in questions:
            for conditioner in conditioners:
                for representation in self.representations:
                    cells.append(Cell(question, conditioner, representation))
        return cells

    def run_side(self, config: ExperimentConfig, questions: Sequence[Question], side_dir: Path) -> None:
        if self.retrieval_eval:
            pairs = sorted({(c.retriever or "", int(c.k or 1)) for c in self.conditions if c.type == "retrieved"})
            write_retrieval_eval(config, questions, pairs, side_dir)
        if self.classifier_eval:
            write_classifier_eval(config, questions, side_dir)


@dataclass(frozen=True)
class YamlExperiment:
    """Resolved YAML experiment file."""

    path: Path
    raw: Mapping[str, Any]
    config: ExperimentConfig
    tasks: tuple[YamlGenerationTask, ...]
    sha256: str

    def write_manifest(self, task: YamlGenerationTask, questions: Sequence[Question]) -> Path:
        """Write a manifest beside a task's prediction cache."""

        paths = experiment_paths(self.config, task.name)
        paths.root.mkdir(parents=True, exist_ok=True)
        path = paths.root / "experiment_manifest.json"
        payload = {
            "version": 1,
            "spec_path": str(self.path),
            "spec_sha256": self.sha256,
            "experiment_name": self.raw.get("name", self.path.stem),
            "run_name": task.name,
            "mode": "smoke" if self.config.smoke else "full",
            "dataset": self.config.dataset,
            "run_tag": self.config.run_tag,
            "dpi": self.config.dpi,
            "models": list(task.models),
            "questions": [q.id for q in questions],
            "question_count": len(questions),
            "representations": list(task.representations),
            "conditions": [
                {
                    "type": c.type,
                    "retriever": c.retriever,
                    "k": c.k,
                    "n_distractors": c.n_distractors,
                    "name_template": c.name_template,
                }
                for c in task.conditions
            ],
            "side_artifacts": {
                "retrieval_eval": task.retrieval_eval,
                "classifier_eval": task.classifier_eval,
            },
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return path


def _task_from_run(raw_run: Mapping[str, Any], config: ExperimentConfig) -> YamlGenerationTask:
    name = _validate_name(str(raw_run.get("name") or "run"))
    raw_cells = raw_run.get("cells") or {}
    if raw_run.get("side_artifacts", {}).get("classifier"):
        raw_cells = raw_cells or {"conditions": [], "representations": []}
    conditions = _condition_specs(raw_cells.get("conditions") or [])
    reps = tuple()
    if raw_cells.get("representations"):
        reps = _validate_representations(raw_cells["representations"])
    elif conditions:
        reps = tuple(config.representations)
    models = tuple(str(x) for x in _as_list(raw_run.get("models"))) or (config.reasoner_spec if conditions else ())
    side = raw_run.get("side_artifacts") or {}
    return YamlGenerationTask(
        name=name,
        models=models,
        selector=QuestionSelector(raw_run.get("questions") or {}),
        conditions=conditions,
        representations=reps,
        retrieval_eval=bool(side.get("retrieval_eval")),
        classifier_eval=bool(side.get("classifier")),
    )


def load_yaml_experiment(path: Path) -> YamlExperiment:
    """Load and validate a YAML experiment spec."""

    source = Path(path).read_text()
    raw = yaml.safe_load(source)
    if not isinstance(raw, Mapping):
        raise SpecError("YAML spec must be a mapping")
    if int(raw.get("version", 0)) != 1:
        raise SpecError("YAML spec version must be 1")
    if not raw.get("runs"):
        raise SpecError("YAML spec must contain at least one run")
    config = _config_from_spec(raw)
    tasks = tuple(_task_from_run(run, config) for run in raw["runs"])
    names = [task.name for task in tasks]
    if len(names) != len(set(names)):
        raise SpecError(f"duplicate run names are not allowed: {names}")
    return YamlExperiment(
        path=Path(path),
        raw=raw,
        config=config,
        tasks=tasks,
        sha256=hashlib.sha256(source.encode()).hexdigest(),
    )
