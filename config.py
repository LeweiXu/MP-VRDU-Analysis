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

# Per-reasoner-size vision-token budget (max pixels per page fed to the image
# processor). Bigger reasoners keep more weights/activations resident, so they
# get a tighter per-page pixel cap to keep the visual sequence inside a 16GB
# V100. Sizes not listed fall back to `ExperimentConfig.max_pixels`. Values are
# `tokens * 28 * 28`, e.g. 768*28*28 -> ~800 vision tokens/page.
MAX_PIXELS_BY_SIZE: dict[str, int] = {
    "8b": 602_112,   # 768*28*28  -> ~800 tok/page
    "32b": 401_408,  # 512*28*28  -> ~520 tok/page
}


def max_pixels_for_spec(spec: str, default: int) -> int:
    """Return the per-page pixel cap for a reasoner spec (size-aware).

    Falls back to `default` (usually `ExperimentConfig.max_pixels`) for the
    smaller sizes that are not in `MAX_PIXELS_BY_SIZE`.
    """

    from models import ModelSpec

    return MAX_PIXELS_BY_SIZE.get(ModelSpec.parse(spec).size, default)


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

    # Cap on how many pixels of a rendered page reach the local VLM image
    # processor. 1280*28*28 = 1,003,520 px -> ~1300 vision tokens/page. Without a
    # cap, 144-DPI pages are ~1.9M px (~2500 tokens each), and a multi-page oracle
    # cell builds a long enough visual sequence that attention OOMs even a 2B
    # model on a 16GB V100 (Volta has no FlashAttention-2, so SDPA falls back to
    # the O(seq^2) math kernel). Tune down for the 8B or docs with many gold pages.
    max_pixels: int = 1_003_520

    paths: ProjectPaths = field(default_factory=ProjectPaths)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bins", tuple(self.bins))
        object.__setattr__(self, "representations", tuple(self.representations))
        if self.smoke:
            object.__setattr__(self, "reasoner_spec", SMOKE_REASONER_SPEC)
            object.__setattr__(self, "max_tokens", min(int(self.max_tokens), SMOKE_MAX_TOKENS))
