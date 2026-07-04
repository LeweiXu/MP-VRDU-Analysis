"""Define root-relative paths and plain-Python experiment configuration.

Purpose:
    Centralises settings that must be identical locally and on Kaya: artifact
    roots, dataset name, smoke/full mode, model specs, condition grids,
    representation rungs, Option-A bins, cost metric, rendering settings, and
    the pre-registered sufficiency margin.

Pipeline role:
    `ProjectPaths` makes the repository self-contained by deriving `.data/`,
    `.cache/`, `results/`, and `envs/` from the repo root. `ExperimentConfig`
    is the immutable object passed into runners and the orchestrator; it is the
    only configuration object later stages should read.

Arguments:
    None. This is an import-only module; callers instantiate dataclasses
    directly, for example `ExperimentConfig(smoke=True)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

DEFAULT_BINS: tuple[str, ...] = ("text_heavy", "in_between", "visual_heavy")
DEFAULT_REASONER_SPEC = "qwen3vl-8b-local"
SMOKE_REASONER_SPEC = "qwen3vl-2b-local"
DEFAULT_MAX_TOKENS = 256
SMOKE_MAX_TOKENS = 64


@dataclass(frozen=True)
class ExperimentConfig:
    """The knobs one experiment run reads. Defaults describe the v3 study."""

    # Dataset is fixed to MMLongBench-Doc for the primary v3 study.
    dataset: str = "mmlongbench"
    smoke: bool = False

    # Reasoner: 8B is the center config for single-model experiments; the scaling
    # sanity appendix runs 2B/32B. Specs are 'family-size-backend' strings.
    reasoner_spec: str = DEFAULT_REASONER_SPEC
    scaling_specs: tuple[str, ...] = (
        "qwen3vl-2b-local",
        "qwen3vl-4b-local",
        "qwen3vl-8b-local",
        "qwen3vl-32b-local",
    )
    judge_spec: str = "stub"

    # Input conditions and their grids.
    conditions: tuple[str, ...] = ("oracle", "retrieved", "full", "buried")
    k_values: tuple[int, ...] = (1, 3, 5)
    burying_levels: tuple[int, ...] = (10, 25, 50)

    # Representation ladder.
    representations: tuple[str, ...] = ("T", "TL", "TLV", "V")

    # Option-A doc-type bins and cost metric for the headline frontier.
    bins: tuple[str, ...] = DEFAULT_BINS
    cost_metric: str = "latency_bs1"

    # Pre-registered sufficiency margin in accuracy points.
    sufficiency_margin: float = 3.0

    # Rendering / sampling knobs.
    dpi: int = 144
    sample: int | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS

    paths: ProjectPaths = field(default_factory=ProjectPaths)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bins", tuple(self.bins))
        object.__setattr__(self, "representations", tuple(self.representations))
        if self.smoke:
            object.__setattr__(self, "reasoner_spec", SMOKE_REASONER_SPEC)
            object.__setattr__(self, "max_tokens", min(int(self.max_tokens), SMOKE_MAX_TOKENS))
