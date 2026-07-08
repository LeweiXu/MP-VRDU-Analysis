"""Artifact-driven judge/build helpers for YAML-generated runs.

Purpose:
    Decouples post-generation work from the exact flags or YAML settings used to
    collect data. The generate phase writes manifests and prediction caches; the
    judge phase can score those artifacts directly, and build can aggregate any
    judged rows it finds.

Pipeline role:
    `cli.judge` uses `judge_manifests()` when running YAML-first workflows.
    `cli.build` can use the scanning helpers to aggregate available judged rows
    without relying on the legacy fixed task registry.

Arguments:
    None. Import-only.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from config import ExperimentConfig
from data.loader import load_longdocurl, load_mmlongbench
from experiments.paths import log
from pipeline.judge import Judge
from pipeline.orchestrator import CachedPrediction, PredictionCache, ResultCache, ResultRow, make_cache_key
from schema import Question


@dataclass(frozen=True)
class ArtifactJudgeStatus:
    """Outcome of judging one manifest directory."""

    run_name: str
    status: str
    results: Path
    scored: int
    error: str = ""


def discover_manifests(config: ExperimentConfig) -> list[Path]:
    """Find YAML experiment manifests under this config's cache root."""

    root = config.paths.cache_dir
    return sorted(root.glob("*/*/experiment_manifest.json"))


def _load_questions_for_manifest(manifest: dict, config: ExperimentConfig) -> dict[str, Question]:
    """Return question_id -> Question for a manifest's dataset."""

    dataset = manifest.get("dataset") or config.dataset
    if dataset == "longdocurl":
        questions = load_longdocurl(data_dir=config.paths.data_dir)
    elif manifest.get("mode") == "smoke":
        from experiments.smoke import load_smoke_questions

        questions = load_smoke_questions(config.paths.data_dir)
    else:
        questions = load_mmlongbench(data_dir=config.paths.data_dir)
    return {question.id: question for question in questions}


def _status_path(manifest_path: Path) -> Path:
    return manifest_path.parent / "judge_status.json"


def _write_status(status: ArtifactJudgeStatus) -> None:
    _status_path(status.results.parent / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "experiment": status.run_name,
                "phase": "judge",
                "status": status.status,
                "results": str(status.results),
                "scored": status.scored,
                "error": status.error,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def judge_manifest(config: ExperimentConfig, manifest_path: Path, judge: Judge) -> ArtifactJudgeStatus:
    """Score all cached predictions beside one manifest."""

    manifest = json.loads(manifest_path.read_text())
    run_name = str(manifest.get("run_name") or manifest_path.parent.name)
    predictions_path = manifest_path.parent / "predictions.jsonl"
    results_path = manifest_path.parent / "results.jsonl"
    if not predictions_path.exists():
        status = ArtifactJudgeStatus(run_name, "failed", results_path, 0, f"missing {predictions_path}")
        _write_status(status)
        return status

    questions = _load_questions_for_manifest(manifest, config)
    prediction_cache = PredictionCache(predictions_path)
    result_cache = ResultCache(results_path)
    dpi = int(manifest.get("dpi") or config.dpi)
    scored = 0
    for record in prediction_cache:
        question = questions.get(record.question_id)
        if question is None:
            raise KeyError(f"{run_name}: question {record.question_id!r} not found in dataset")
        prediction = record.as_prediction()
        cache_key = make_cache_key(
            question,
            record.condition,
            record.representation,
            record.model_spec,
            judge.spec,
            dpi,
        )
        if result_cache.get(cache_key) is not None:
            continue
        score = judge.score(question, prediction)
        row = ResultRow(
            cache_key=cache_key,
            question_id=question.id,
            doc_id=question.doc_id,
            doc_type=question.doc_type,
            hop=question.hop,
            is_unanswerable=question.is_unanswerable,
            evidence_sources=question.evidence_sources,
            condition=record.condition,
            provenance=record.provenance,
            page_indices=record.page_indices,
            representation=record.representation,
            model_spec=record.model_spec,
            judge_spec=score.judge_spec or judge.spec,
            answer=prediction.text,
            input_text_tokens=prediction.input_text_tokens,
            input_visual_tokens=prediction.input_visual_tokens,
            output_tokens=prediction.output_tokens,
            latency_s=prediction.latency_s,
            score=score.value,
            correct=score.correct,
            abstained=score.abstained,
            metadata={
                "note": record.note,
                "source_dataset": question.raw_fields.get("source_dataset", manifest.get("dataset", config.dataset)),
                "manifest": str(manifest_path),
            },
        )
        result_cache.put(row)
        scored += 1
    status = ArtifactJudgeStatus(run_name, "success", results_path, scored)
    _write_status(status)
    log.info("[judge] %s: scored %d prediction(s) -> %s", run_name, scored, results_path)
    return status


def judge_manifests(
    config: ExperimentConfig,
    judge: Judge,
    *,
    manifests: Sequence[Path] | None = None,
) -> list[ArtifactJudgeStatus]:
    """Score every discovered manifest under a cache root."""

    paths = list(manifests) if manifests is not None else discover_manifests(config)
    return [judge_manifest(config, path, judge) for path in paths]


def iter_result_files(config: ExperimentConfig) -> Iterable[Path]:
    """Yield judged result files under this config's cache root."""

    yield from sorted(config.paths.cache_dir.glob("*/*/results.jsonl"))


def iter_side_files(config: ExperimentConfig, filename: str) -> Iterable[Path]:
    """Yield side artifact files under this config's cache root."""

    yield from sorted(config.paths.cache_dir.glob(f"*/*/{filename}"))
