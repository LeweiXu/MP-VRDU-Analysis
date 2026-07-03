"""Stage 0 skeleton and Kaya contract tests."""

from __future__ import annotations

import importlib
import json
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
    "kaya",
    "kaya.kaya",
    "kaya.download_hf",
]


def test_tree_imports() -> None:
    """Every Stage 0 module imports without triggering optional dependencies."""
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        assert module.__doc__


def test_kaya_config_paths_are_root_relative() -> None:
    """Kaya config places all remote artifacts under one mirror root."""
    from kaya.kaya import load_config

    config = load_config(ROOT / "kaya/config.json")

    assert config.ssh_alias == "kaya"
    assert config.remote_root == "/group/ems036/lxu/mpvrdu"
    assert config.remote_path("cache") == f"{config.remote_root}/.cache"
    assert config.remote_path("env") == f"{config.remote_root}/envs/mpvrdu"
    assert config.remote_path("data") == f"{config.remote_root}/.data"
    assert config.remote_path("results") == f"{config.remote_root}/results"
    assert config.remote_path("logs") == f"{config.remote_root}/logs"
    assert config.raw["hf"]["disable_xet"] is True
    assert config.raw["hf"]["max_workers"] == 1
    assert "account" in config.raw["slurm"]
    assert "qos" in config.raw["slurm"]


def test_kaya_config_excludes_heavy_dirs() -> None:
    """Rsync excludes protect machine-local artifacts from cross-machine copies."""
    config = json.loads((ROOT / "kaya/config.json").read_text())
    excludes = set(config["rsync_excludes"])
    for excluded in [".git/", ".cache/", ".data/", "envs/", "results/", "logs/"]:
        assert excluded in excludes

    assert not (ROOT / "scripts/kaya").exists()
    assert (ROOT / "kaya/KAYA_AGENT_GUIDE.md").is_file()
    assert (ROOT / "kaya/KAYA_USER_GUIDE.md").is_file()


def test_data_package_is_code_not_artifacts() -> None:
    """The importable data package is separate from the ignored .data artifact root."""
    assert (ROOT / "data/__init__.py").is_file()
    assert (ROOT / "data/loader.py").is_file()
    assert (ROOT / "data/render.py").is_file()
    assert (ROOT / ".data").is_dir()
