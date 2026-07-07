"""Test YAML-driven generation specs and artifact judging.

Purpose:
    Protects the YAML-first experiment interface: cell-grid expansion,
    non-additive representations, manifest writing, and post-generation judging
    without matching the original generate flags.

Test role:
    Uses tiny fixture PDFs and fake reasoner/judge paths, so it validates the
    orchestration and cache semantics without loading real models.

Arguments:
    None. Run with `python -m pytest tests/test_yaml_experiments.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import ExperimentConfig, ProjectPaths
from data.loader import load_mmlongbench
from experiments.artifacts import judge_manifests
from experiments.driver import run_generate_tasks
from experiments.paths import experiment_paths
from experiments.yaml_spec import load_yaml_experiment
from models.payload import ModelInput
from pipeline.judge import Judge
from pipeline.reasoner import Reasoner
from schema import Prediction, Question, Score


class TinyReasoner(Reasoner):
    spec = "qwen3vl-2b-local"

    def answer(self, question: Question, model_input: ModelInput) -> Prediction:
        return Prediction(
            text=f"{question.gold_answer} generated",
            model_spec=self.spec,
            input_text_tokens=sum(len(part.text.split()) for part in model_input.text_parts),
            input_visual_tokens=7 * len(model_input.image_parts),
            output_tokens=2,
            latency_s=0.01,
        )


class TinyJudge(Judge):
    spec = "tiny-judge"

    def score(self, question: Question, prediction: Prediction) -> Score:
        correct = question.gold_answer in prediction.text
        return Score(value=1.0 if correct else 0.0, correct=correct, judge_spec=self.spec)


def write_pdf(path: Path, pages: list[str]) -> None:
    import fitz

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def make_fixture(tmp_path: Path) -> ProjectPaths:
    data_dir = tmp_path / ".data"
    root = data_dir / "mmlongbench"
    (root / "data").mkdir(parents=True)
    (root / "documents").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "doc_id": "doc-a.pdf",
                "doc_type": "Academic paper",
                "question": "What is alpha?",
                "answer": "alpha",
                "evidence_pages": "[1]",
                "evidence_sources": "['Pure-text (Plain-text)']",
                "answer_format": "String",
            }
        ]
    ).to_parquet(root / "data" / "data.parquet")
    write_pdf(root / "documents" / "doc-a.pdf", ["cover", "alpha evidence"])
    return ProjectPaths(root=tmp_path, data_dir=data_dir, results_dir=tmp_path / "results", cache_dir=tmp_path / "results" / "cache")


def write_spec(tmp_path: Path, paths: ProjectPaths) -> Path:
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        """
version: 1
name: yaml-test
run_tag: ytest
mode: full
dataset: mmlongbench
config:
  per_bin_sample:
  max_tokens: 64
  representations: [TV, LV]
runs:
  - name: yaml_run
    questions:
      limit: 1
    models: [qwen3vl-2b-local]
    cells:
      conditions:
        - type: oracle
      representations: [TV, LV]
""".strip()
        + "\n"
    )
    return spec


def test_yaml_expands_non_additive_representations_and_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    import experiments.driver as driver
    import pipeline.representation as representation

    paths = make_fixture(tmp_path)
    spec_path = write_spec(tmp_path, paths)
    experiment = load_yaml_experiment(spec_path)
    config = ExperimentConfig(**{**experiment.config.__dict__, "paths": paths})
    object.__setattr__(experiment, "config", config)
    questions = load_mmlongbench(paths.data_dir)

    monkeypatch.setattr(driver, "reasoner_for", lambda spec, config=None: TinyReasoner())
    monkeypatch.setattr(representation, "text_channel", lambda pages: tuple(page.text for page in pages))
    monkeypatch.setattr(representation, "layout_channel", lambda pages: tuple("{}" for _ in pages))

    statuses = run_generate_tasks(config, experiment.tasks, questions, before_task=experiment.write_manifest)

    assert [status.status for status in statuses] == ["success"]
    paths_for_run = experiment_paths(config, "yaml_run")
    manifest = json.loads((paths_for_run.root / "experiment_manifest.json").read_text())
    assert manifest["representations"] == ["TV", "LV"]
    predictions = paths_for_run.predictions.read_text().splitlines()
    assert len(predictions) == 2
    reps = {json.loads(line)["representation"] for line in predictions}
    assert reps == {"TV", "LV"}


def test_artifact_judge_scores_manifest_without_generation_flags(tmp_path: Path, monkeypatch) -> None:
    import experiments.driver as driver
    import pipeline.representation as representation

    paths = make_fixture(tmp_path)
    spec_path = write_spec(tmp_path, paths)
    experiment = load_yaml_experiment(spec_path)
    config = ExperimentConfig(**{**experiment.config.__dict__, "paths": paths})
    object.__setattr__(experiment, "config", config)
    questions = load_mmlongbench(paths.data_dir)

    monkeypatch.setattr(driver, "reasoner_for", lambda spec, config=None: TinyReasoner())
    monkeypatch.setattr(representation, "text_channel", lambda pages: tuple(page.text for page in pages))
    monkeypatch.setattr(representation, "layout_channel", lambda pages: tuple("{}" for _ in pages))
    run_generate_tasks(config, experiment.tasks, questions, before_task=experiment.write_manifest)

    statuses = judge_manifests(config, TinyJudge())

    assert [status.status for status in statuses] == ["success"]
    results = experiment_paths(config, "yaml_run").results.read_text().splitlines()
    assert len(results) == 2
    assert all(json.loads(line)["judge_spec"] == "tiny-judge" for line in results)
