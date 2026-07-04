"""Test repository skeleton imports and basic Kaya configuration contracts.

Purpose:
    Verifies that placeholder packages import, required project files exist, and
    Kaya configuration contains the expected root-relative paths and exclusions.

Test role:
    Protects the initial scaffold assumptions that later staged work builds on.

Arguments:
    None. Run with `python -m pytest tests/test_skeleton.py`.
"""

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
    "kaya.setup_env",
    "kaya.prestage",
    "kaya.gpu_test",
    "kaya.run_probe",
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
    assert config.raw["hf"]["max_workers"] == 8
    assert config.raw["secrets"]["env_file"] == ".env"
    assert "HF_TOKEN" in config.raw["secrets"]["forward"]
    assert "account" in config.raw["slurm"]
    assert "qos" in config.raw["slurm"]
    assert "BAAI/bge-small-en-v1.5" in config.raw["retrieval_models"]["text"]
    assert "vidore/colqwen2.5-v0.2" in config.raw["retrieval_models"]["vision"]
    assert config.raw["tool_caches"]["paddleocr"] is True
    assert config.raw["tool_caches"]["docling"] is True


def test_kaya_config_excludes_heavy_dirs() -> None:
    """Rsync excludes protect machine-local artifacts from cross-machine copies."""
    config = json.loads((ROOT / "kaya/config.json").read_text())
    excludes = set(config["rsync_excludes"])
    for excluded in [".git/", ".env", ".cache/", ".data/", "envs/", "results/", "logs/"]:
        assert excluded in excludes

    assert not (ROOT / "scripts/kaya").exists()
    assert (ROOT / "kaya/KAYA_AGENT_GUIDE.md").is_file()
    assert (ROOT / "kaya/KAYA_USER_GUIDE.md").is_file()


def test_kaya_python_header_parsing(tmp_path: Path) -> None:
    """Runnable Kaya Python files can declare default execution settings."""
    from kaya.kaya import parse_kaya_header, resolve_run_settings

    script = tmp_path / "probe.py"
    script.write_text(
        "\n".join(
            [
                '"""demo"""',
                "# kaya: target=gpu",
                "# kaya: env=true",
                "# kaya: offline=true",
                "# kaya: job-name=demo_job",
                "print('ok')",
            ]
        )
    )

    assert parse_kaya_header(script)["target"] == "gpu"
    settings = resolve_run_settings(script)
    assert settings.target == "gpu"
    assert settings.activate_env is True
    assert settings.offline is True
    assert settings.job_name == "demo_job"

    override = resolve_run_settings(script, target_override="login", offline_override=False)
    assert override.target == "login"
    assert override.offline is False


def test_data_package_is_code_not_artifacts() -> None:
    """The importable data package is separate from the ignored .data artifact root."""
    assert (ROOT / "data/__init__.py").is_file()
    assert (ROOT / "data/loader.py").is_file()
    assert (ROOT / "data/render.py").is_file()
    assert (ROOT / ".data").is_dir()


def test_paddlex_is_pinned_for_paddleocr_api() -> None:
    """PaddleOCR 3.1 uses the PaddleX 3.1 predictor-option constructor."""
    requirements = (ROOT / "requirements.txt").read_text()
    assert "paddleocr==3.1.0" in requirements
    assert "paddlex[ie,multimodal,ocr]>=3.1.0,<3.2.0" in requirements
