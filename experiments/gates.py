"""Section-2 gate utilities for frontier, agreement, and classifier checks.

Purpose:
    Implements the three go/no-go gates that sit before the full paper runs are
    trusted: F1 frontier divergence, F2 judge-human agreement, and F3 classifier
    feasibility. The functions here are deliberately pure and artifact-oriented:
    they read/write CSV or JSONL records, compute deterministic predicates, and
    leave expensive generation/classification to callers.

Pipeline role:
    `scripts.gates` exposes these helpers for real runs, while tests exercise the
    predicates and samplers without loading models. Keeping the gate logic here
    makes the human checkpoints reproducible from cached rows and side artifacts.

Arguments:
    None. Import callers pass result rows, questions, CSV paths, or classifier
    record dictionaries to the public functions.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from config import DEFAULT_BINS, ExperimentConfig
from data.binning import doc_type_bin
from pipeline.orchestrator import ResultRow
from schema import Question


AGREEMENT_LABELS: tuple[str, ...] = ("correct", "incorrect", "abstained")


@dataclass(frozen=True)
class GateResult:
    """One machine-readable go/no-go result."""

    gate: str
    passed: bool
    metric: float
    threshold: float
    status: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""

        data = asdict(self)
        data["details"] = dict(self.details)
        return data


def write_gate_json(result: GateResult, path: Path) -> Path:
    """Write a gate result as stable JSON and return the path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n")
    return path


# -- F1: frontier divergence -------------------------------------------------


def load_table1_frontiers(path: Path) -> dict[str, str]:
    """Load `bin -> frontier` values from a Table-1 CSV."""

    frontiers: dict[str, str] = {}
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "bin" not in (reader.fieldnames or ()) or "frontier" not in (reader.fieldnames or ()):
            raise ValueError(f"{path} must contain 'bin' and 'frontier' columns")
        for row in reader:
            bin_name = str(row.get("bin", "")).strip()
            if bin_name:
                frontiers[bin_name] = str(row.get("frontier", "")).strip()
    return frontiers


def frontier_divergence_gate(
    frontiers: Mapping[str, str],
    *,
    bins: Sequence[str] = DEFAULT_BINS,
    min_distinct: int = 2,
) -> GateResult:
    """Return Go when at least two configured bins have different frontiers."""

    selected = {bin_name: str(frontiers.get(bin_name, "")).strip() for bin_name in bins}
    missing = [bin_name for bin_name, frontier in selected.items() if not frontier]
    if missing:
        raise ValueError(f"missing frontier value for bin(s): {', '.join(missing)}")

    distinct = sorted(set(selected.values()))
    passed = len(distinct) >= int(min_distinct)
    return GateResult(
        gate="F1_frontier_divergence",
        passed=passed,
        metric=float(len(distinct)),
        threshold=float(min_distinct),
        status="go" if passed else "no_go",
        message=(
            "Go: doc-type bins do not share one sufficiency frontier"
            if passed
            else "No-go: all configured doc-type bins share one sufficiency frontier"
        ),
        details={"frontiers": selected, "distinct_frontiers": distinct},
    )


# -- Shared samplers ---------------------------------------------------------


def question_type_label(question: Question) -> str:
    """Return the best available question-type label for stratification."""

    for key in ("question_type", "task_tag", "question_category"):
        value = question.raw_fields.get(key)
        if value not in (None, ""):
            return str(value)
    return question.hop


def _shuffle_groups(groups: Mapping[tuple[str, ...], Sequence[Question]], seed: int) -> dict[tuple[str, ...], list[Question]]:
    """Return deterministically shuffled question groups."""

    rng = random.Random(seed)
    shuffled: dict[tuple[str, ...], list[Question]] = {}
    for key, values in groups.items():
        items = sorted(values, key=lambda q: q.id)
        rng.shuffle(items)
        shuffled[key] = items
    return shuffled


def stratified_question_sample(
    questions: Sequence[Question],
    *,
    n: int,
    seed: int = 0,
) -> list[Question]:
    """Sample questions over doc_type x question_type, covering each non-empty cell."""

    if n <= 0:
        raise ValueError("n must be positive")
    groups: dict[tuple[str, str], list[Question]] = defaultdict(list)
    for question in questions:
        groups[(question.doc_type, question_type_label(question))].append(question)
    if not groups:
        return []

    remaining = _shuffle_groups(groups, seed)
    keys = sorted(remaining)
    selected: list[Question] = []
    target = min(int(n), sum(len(values) for values in remaining.values()))

    # First pass: one item from every cell, when the requested sample is large
    # enough. This is the coverage guarantee required by the F2 gate.
    if target >= len(keys):
        for key in keys:
            selected.append(remaining[key].pop(0))

    # Fill the rest in deterministic round-robin order to avoid one large stratum
    # consuming the sample after the coverage pass.
    while len(selected) < target:
        progressed = False
        for key in keys:
            if len(selected) >= target:
                break
            if remaining[key]:
                selected.append(remaining[key].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def classifier_pilot_sample(
    questions: Sequence[Question],
    *,
    n_docs: int = 100,
    seed: int = 0,
) -> list[Question]:
    """Sample one representative question for `n_docs` distinct documents."""

    if n_docs <= 0:
        raise ValueError("n_docs must be positive")
    by_doc: dict[str, Question] = {}
    for question in sorted(questions, key=lambda q: (q.doc_id, q.id)):
        by_doc.setdefault(question.doc_id, question)
    if len(by_doc) < n_docs:
        raise ValueError(f"requested {n_docs} documents but only {len(by_doc)} are available")

    groups: dict[tuple[str, ...], list[Question]] = defaultdict(list)
    for question in by_doc.values():
        groups[(question.doc_type,)].append(question)
    remaining = _shuffle_groups(groups, seed)
    keys = sorted(remaining)
    selected: list[Question] = []
    while len(selected) < n_docs:
        for key in keys:
            if len(selected) >= n_docs:
                break
            if remaining[key]:
                selected.append(remaining[key].pop(0))
    return selected


# -- F2: judge-human agreement ----------------------------------------------


def result_row_label(row: ResultRow) -> str:
    """Return the agreement label implied by a judged result row."""

    if row.abstained:
        return "abstained"
    return "correct" if row.correct else "incorrect"


def normalise_agreement_label(value: str) -> str:
    """Normalise a human or judge label to the agreement label set."""

    label = " ".join(str(value).strip().replace("_", " ").split()).casefold()
    aliases = {
        "correct": "correct",
        "right": "correct",
        "yes": "correct",
        "incorrect": "incorrect",
        "wrong": "incorrect",
        "no": "incorrect",
        "abstained": "abstained",
        "abstain": "abstained",
        "refusal": "abstained",
    }
    try:
        return aliases[label]
    except KeyError as exc:
        raise ValueError(f"unknown agreement label {value!r}") from exc


def cohen_kappa(
    judge_labels: Sequence[str],
    human_labels: Sequence[str],
    *,
    labels: Sequence[str] = AGREEMENT_LABELS,
) -> float:
    """Compute Cohen's kappa for two equal-length label sequences."""

    if len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels and human_labels must have the same length")
    if not judge_labels:
        raise ValueError("at least one labelled item is required")

    normal_judge = [normalise_agreement_label(label) for label in judge_labels]
    normal_human = [normalise_agreement_label(label) for label in human_labels]
    n = len(normal_judge)
    observed = sum(a == b for a, b in zip(normal_judge, normal_human, strict=True)) / n
    judge_counts = Counter(normal_judge)
    human_counts = Counter(normal_human)
    expected = sum((judge_counts[label] / n) * (human_counts[label] / n) for label in labels)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def agreement_gate(kappa: float, *, threshold: float = 0.75, n_items: int = 0) -> GateResult:
    """Return Go when judge-human Cohen's kappa reaches the configured threshold."""

    passed = float(kappa) >= float(threshold)
    return GateResult(
        gate="F2_judge_human_agreement",
        passed=passed,
        metric=float(kappa),
        threshold=float(threshold),
        status="go" if passed else "no_go",
        message=(
            "Go: judge-human agreement meets the pre-registered threshold"
            if passed
            else "No-go: judge-human agreement is below the pre-registered threshold"
        ),
        details={"n_items": int(n_items)},
    )


def _row_filter(
    row: ResultRow,
    *,
    condition: str | None,
    representation: str | None,
    model_spec: str | None,
) -> bool:
    """Return whether a result row belongs in the agreement sampling frame."""

    if condition is not None and row.condition != condition:
        return False
    if representation is not None and row.representation != representation:
        return False
    if model_spec is not None and row.model_spec != model_spec:
        return False
    return True


def agreement_sheet_rows(
    questions: Sequence[Question],
    rows: Sequence[ResultRow],
    *,
    n: int = 200,
    seed: int = 0,
    condition: str | None = "oracle",
    representation: str | None = "TLV",
    model_spec: str | None = None,
) -> list[dict[str, str]]:
    """Build rows for the human-labelling agreement sheet."""

    question_by_id = {question.id: question for question in questions}
    result_by_question: dict[str, ResultRow] = {}
    for row in rows:
        if row.question_id not in question_by_id:
            continue
        if not _row_filter(row, condition=condition, representation=representation, model_spec=model_spec):
            continue
        result_by_question.setdefault(row.question_id, row)

    eligible = [question for question in questions if question.id in result_by_question]
    sampled = stratified_question_sample(eligible, n=n, seed=seed)
    sheet: list[dict[str, str]] = []
    for question in sampled:
        row = result_by_question[question.id]
        sheet.append(
            {
                "question_id": question.id,
                "doc_id": question.doc_id,
                "doc_type": question.doc_type,
                "question_type": question_type_label(question),
                "question": question.question,
                "gold_answer": question.gold_answer,
                "model_spec": row.model_spec,
                "condition": row.condition,
                "representation": row.representation,
                "model_answer": row.answer,
                "judge_label": result_row_label(row),
                "human_label": "",
                "notes": "",
            }
        )
    return sheet


def render_agreement_packet(
    config: "ExperimentConfig",
    sheet_rows: Sequence[Mapping[str, Any]],
    out_dir: Path,
    *,
    task: str = "G1_sufficiency",
) -> Path:
    """Render the sampled cells into a viewing packet for human labelling.

    The CSV sheet is text-only, so a labeller can't see the document. This joins
    each sampled row back to its cached cell (by question/condition/representation/
    model_spec) and renders the fed pages + a scrollable markdown packet under
    `out_dir`, in the same order as the CSV. The human reads the packet in VSCode
    and fills `human_label` in the CSV (matched by question_id).
    """

    # Imported here so the pure gate predicates stay import-light for tests.
    from experiments.inspect import items_by_cell, write_packet

    by_cell = items_by_cell(config, task)
    items = []
    for row in sheet_rows:
        key = (row["question_id"], row["condition"], row["representation"], row["model_spec"])
        item = by_cell.get(key)
        if item is not None:
            items.append(item)
    return write_packet(
        items,
        Path(out_dir),
        config,
        title="F2 judge-vs-human agreement — label each in the CSV's human_label column",
        packet_name="agreement_view.md",
    )


def write_csv_records(records: Sequence[Mapping[str, Any]], path: Path) -> Path:
    """Write dictionaries to CSV using stable field order from the first record."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("")
        return path
    fieldnames = list(records[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return path


def read_agreement_labels(path: Path) -> tuple[list[str], list[str]]:
    """Read non-empty judge/human label pairs from an agreement sheet CSV."""

    judge_labels: list[str] = []
    human_labels: list[str] = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            human = str(row.get("human_label", "")).strip()
            if not human:
                continue
            judge = str(row.get("judge_label", "")).strip()
            judge_labels.append(normalise_agreement_label(judge))
            human_labels.append(normalise_agreement_label(human))
    if not judge_labels:
        raise ValueError(f"{path} has no completed human_label rows")
    return judge_labels, human_labels


def score_agreement_sheet(path: Path, *, threshold: float = 0.75) -> GateResult:
    """Compute kappa from a completed agreement sheet and return the gate result."""

    judge_labels, human_labels = read_agreement_labels(path)
    kappa = cohen_kappa(judge_labels, human_labels)
    result = agreement_gate(kappa, threshold=threshold, n_items=len(judge_labels))
    counts = {
        "judge": dict(Counter(judge_labels)),
        "human": dict(Counter(human_labels)),
    }
    return GateResult(
        gate=result.gate,
        passed=result.passed,
        metric=result.metric,
        threshold=result.threshold,
        status=result.status,
        message=result.message,
        details={**dict(result.details), "label_counts": counts},
    )


# -- F3: classifier feasibility ---------------------------------------------


def classifier_gate(accuracy: float, *, threshold: float = 0.70, n_docs: int = 0) -> GateResult:
    """Return Go when classifier top-1 bin accuracy reaches the threshold."""

    passed = float(accuracy) >= float(threshold)
    return GateResult(
        gate="F3_classifier_feasibility",
        passed=passed,
        metric=float(accuracy),
        threshold=float(threshold),
        status="go" if passed else "no_go",
        message=(
            "Go: predicted routing may use the doc-type classifier"
            if passed
            else "No-go: scope RQ3 to oracle routing or upgrade the classifier"
        ),
        details={"n_docs": int(n_docs)},
    )


def classifier_record_from_prediction(question: Question, prediction: Any, *, classifier_name: str) -> dict[str, Any]:
    """Return one CSV/JSON-ready classifier pilot record."""

    gold_bin = doc_type_bin(question.doc_type)
    predicted_bin = str(getattr(prediction, "bin", "") or "")
    predicted_doc_type = str(getattr(prediction, "doc_type", ""))
    return {
        "doc_id": question.doc_id,
        "question_id": question.id,
        "gold_doc_type": question.doc_type,
        "predicted_doc_type": predicted_doc_type,
        "gold_bin": gold_bin,
        "predicted_bin": predicted_bin,
        "correct_doc_type": predicted_doc_type == question.doc_type,
        "correct_bin": predicted_bin == gold_bin,
        "confidence": float(getattr(prediction, "confidence", 0.0)),
        "latency_s": float(getattr(prediction, "latency_s", 0.0)),
        "classifier": classifier_name,
        "raw_text": str(getattr(prediction, "raw_text", "")),
    }


def run_classifier_pilot(
    questions: Sequence[Question],
    classifier: Any,
    *,
    n_docs: int = 100,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Run a classifier on a deterministic distinct-document pilot sample."""

    sampled = classifier_pilot_sample(questions, n_docs=n_docs, seed=seed)
    classifier_name = str(getattr(classifier, "name", classifier.__class__.__name__))
    records: list[dict[str, Any]] = []
    for question in sampled:
        prediction = classifier.classify(question)
        records.append(classifier_record_from_prediction(question, prediction, classifier_name=classifier_name))
    return records


def _truthy(value: Any) -> bool:
    """Return a permissive boolean parse for CSV/JSON values."""

    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def classifier_accuracy(records: Iterable[Mapping[str, Any]]) -> tuple[float, int]:
    """Return top-1 Option-A bin accuracy and document count from records."""

    materialized = list(records)
    if not materialized:
        raise ValueError("at least one classifier record is required")
    correct = sum(1 for record in materialized if _truthy(record.get("correct_bin", False)))
    doc_count = len({str(record.get("doc_id", "")) for record in materialized if record.get("doc_id")})
    return correct / len(materialized), doc_count or len(materialized)


def score_classifier_records(
    records: Sequence[Mapping[str, Any]],
    *,
    threshold: float = 0.70,
) -> GateResult:
    """Score classifier records and return the F3 gate result."""

    accuracy, n_docs = classifier_accuracy(records)
    result = classifier_gate(accuracy, threshold=threshold, n_docs=n_docs)
    return GateResult(
        gate=result.gate,
        passed=result.passed,
        metric=result.metric,
        threshold=result.threshold,
        status=result.status,
        message=result.message,
        details={
            **dict(result.details),
            "correct_docs": sum(1 for record in records if _truthy(record.get("correct_bin", False))),
        },
    )


def read_classifier_records(path: Path) -> list[dict[str, Any]]:
    """Read classifier records from CSV or JSONL artifacts."""

    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
