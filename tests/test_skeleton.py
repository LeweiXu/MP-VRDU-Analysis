"""Stage 0 skeleton and Kaya contract tests."""

from __future__ import annotations

import importlib
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MODULES = [
    "config",
    "schema",
    "data",
    "data.loader",
    "data.render",
    "pipeline",
    "pipeline.conditioner",
    "pipeline.representation",
    "pipeline.reasoner",
    "pipeline.judge",
    "pipeline.orchestrator",
    "models",
    "models.payload",
    "models.local_vlm",
    "models.api_vlm",
    "covariates",
    "covariates.retriever",
    "covariates.classifier",
    "tools",
    "tools.text",
    "tools.layout",
    "tools.visual",
    "metrics",
    "metrics.accuracy",
    "metrics.retrieval",
    "metrics.abstention",
    "metrics.cost",
    "metrics.frontier",
    "experiments",
    "experiments.runner",
    "experiments.tables",
    "cli",
    "cli.run_probe",
    "cli.run_experiment",
    "cli.build_tables",
]


def test_tree_imports() -> None:
    """Every Stage 0 module imports without triggering optional dependencies."""
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        assert module.__doc__


def test_kaya_env_paths_are_root_relative() -> None:
    """Kaya env defaults place all artifacts under the remote repo mirror."""
    command = (
        "source scripts/kaya/env.sh; "
        "printf '%s\\n' "
        "\"MPVRDU_ROOT=$MPVRDU_ROOT\" "
        "\"KAYA_REMOTE_DIR=$KAYA_REMOTE_DIR\" "
        "\"HF_HOME=$HF_HOME\" "
        "\"KAYA_ENV=$KAYA_ENV\" "
        "\"KAYA_DATA_DIR=$KAYA_DATA_DIR\" "
        "\"KAYA_RESULTS_DIR=$KAYA_RESULTS_DIR\" "
        "\"KAYA_LOGS_DIR=$KAYA_LOGS_DIR\""
    )
    clean_env = {
        "HOME": os.environ.get("HOME", ""),
        "PATH": os.environ.get("PATH", ""),
        "KAYA_USER": "user",
        "KAYA_PROJECT": "project",
    }
    result = subprocess.run(
        ["bash", "-lc", command],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        env=clean_env,
    )
    values = dict(line.split("=", 1) for line in result.stdout.splitlines())

    assert values["MPVRDU_ROOT"] == str(ROOT)
    assert values["KAYA_REMOTE_DIR"] == "/group/project/user/mpvrdu"
    assert values["HF_HOME"] == f'{values["KAYA_REMOTE_DIR"]}/.cache'
    assert values["KAYA_ENV"] == f'{values["KAYA_REMOTE_DIR"]}/envs/mpvrdu'
    assert values["KAYA_DATA_DIR"] == f'{values["KAYA_REMOTE_DIR"]}/.data'
    assert values["KAYA_RESULTS_DIR"] == f'{values["KAYA_REMOTE_DIR"]}/results'
    assert values["KAYA_LOGS_DIR"] == f'{values["KAYA_REMOTE_DIR"]}/logs'


def test_kaya_sync_excludes_heavy_dirs() -> None:
    """Rsync excludes protect machine-local artifacts from cross-machine copies."""
    sync_script = (ROOT / "scripts/kaya/sync_kaya.sh").read_text()
    for excluded in ['".git/"', '".cache/"', '".data/"', '"envs/"', '"results/"', '"logs/"']:
        assert f"--exclude {excluded}" in sync_script

    assert 'KAYA_RESULTS_DIR/" "$ROOT/results/"' in sync_script
    assert 'KAYA_LOGS_DIR/" "$ROOT/logs/"' in sync_script


def test_data_package_is_code_not_artifacts() -> None:
    """The importable data package is separate from the ignored .data artifact root."""
    assert (ROOT / "data/__init__.py").is_file()
    assert (ROOT / "data/loader.py").is_file()
    assert (ROOT / "data/render.py").is_file()
    assert (ROOT / ".data").is_dir()
