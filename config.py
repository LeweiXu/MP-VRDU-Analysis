"""Experiment configuration entry point for root-relative project settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def project_root(start: Path | None = None) -> Path:
    """Return the repository root by walking up to `docs/implementation_plan.md`."""

    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "docs/implementation_plan.md").is_file():
            return candidate
    raise FileNotFoundError("could not locate repository root")


ROOT = project_root()


@dataclass(frozen=True)
class ProjectPaths:
    """Root-relative artifact paths shared by local and Kaya execution."""

    root: Path = ROOT
    data_dir: Path = ROOT / ".data"
    hf_home: Path = ROOT / ".cache"
    results_dir: Path = ROOT / "results"
    cache_dir: Path = ROOT / "results" / "cache"
    env_dir: Path = ROOT / "envs"


DEFAULT_PATHS = ProjectPaths()
