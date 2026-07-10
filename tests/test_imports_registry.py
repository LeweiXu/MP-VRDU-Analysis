"""Every task, backend, retriever, parser, and table builder is importable and
discoverable through its registry."""

from __future__ import annotations

import importlib

import pytest

from conftest import require

SPINE_MODULES = [
    "config", "schema",
    "data.loader", "data.binning", "data.annotations", "data.render",
    "tools.text", "tools.parser", "tools.visual",
    "retrievers.text", "retrievers.vision", "retrievers.joint",
    "models.qwen3vl", "models.internvl", "models.classifier", "models.payload",
    "pipeline.conditioner", "pipeline.representation", "pipeline.reasoner",
    "pipeline.judge", "pipeline.orchestrator",
    "scoring.accuracy", "scoring.cost", "scoring.frontier", "scoring.retrieval",
    "scoring.abstention", "scoring.agreement",
    "experiments.registry",
    "experiments.tasks.base",
    "experiments.tasks.G1_oracle_ladder", "experiments.tasks.G2_retrieval",
    "experiments.tasks.G3_hallucination",
    "experiments.engine.driver", "experiments.engine.side_artifacts",
    "experiments.engine.artifacts", "experiments.engine.paths",
    "experiments.corpus.resolve", "experiments.corpus.smoke",
    "experiments.corpus.yaml_spec",
    "reporting.build",
    "reporting.tables.headline", "reporting.tables.parser",
    "reporting.tables.resolution", "reporting.tables.matched_cross",
    "reporting.tables.kdepth", "reporting.tables.retrieval_accuracy",
    "reporting.tables.hallucination", "reporting.tables.routing",
    "reporting.tables.scale", "reporting.tables.composition",
]

EXPECTED_TASKS = {
    "G1_oracle_ladder", "G2_retrieval", "G3_hallucination",
}


@pytest.mark.parametrize("mod", SPINE_MODULES)
def test_spine_module_imports(mod: str) -> None:
    importlib.import_module(mod)


def test_registry_lists_the_three_tasks() -> None:
    # The registry maps task name -> task collection. All three G-tasks resolve;
    # no RQ/table numbers, only the G[num]_[name] handles.
    get = require("experiments.registry", "get_task")
    for name in EXPECTED_TASKS:
        assert get(name) is not None


def test_registry_has_no_legacy_task_names() -> None:
    reg = importlib.import_module("experiments.registry")
    names = set(getattr(reg, "TASKS", {}) or {})
    if not names:
        pytest.fail("experiments.registry.TASKS not populated yet (v4 stub)")
    assert names == EXPECTED_TASKS, f"unexpected task set: {names}"
