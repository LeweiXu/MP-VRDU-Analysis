"""Run knobs shared across the pipeline: filesystem paths, model specs, named
visual-resolution presets, and sampling defaults."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path


def project_root(start: Path | None = None) -> Path:
    """Return the repository root by walking up to the repo's `README.md`."""

    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "README.md").is_file() and (candidate / "config.py").is_file():
            return candidate
    raise FileNotFoundError("could not locate repository root")


ROOT = project_root()

# Cache namespace. Every cached cell lives under `results/cache/<CACHE_VERSION>/`
# so a cell from an earlier mechanism can never be read back as if it were a
# current one. Bump this string whenever a change would make old cached rows
# semantically wrong for a new run.
CACHE_VERSION = "v4"


@dataclass(frozen=True)
class ProjectPaths:
    """Root-relative artifact paths shared by local and Kaya execution."""

    root: Path = ROOT
    data_dir: Path = ROOT / ".data"
    hf_home: Path = ROOT / ".cache"
    results_dir: Path = ROOT / "results"
    cache_dir: Path = ROOT / "results" / "cache" / CACHE_VERSION
    env_dir: Path = ROOT / "envs"


DEFAULT_PATHS = ProjectPaths()

# Manual-annotation modality bins, ordered text -> visual. The bin axis is the
# whole thesis and is labelled by hand (data.annotations), not derived from the
# native doc_type.
DEFAULT_BINS: tuple[str, ...] = ("text-dominant", "mixed-modality", "visual-dominant")

DEFAULT_REASONER_SPEC = "qwen3vl-8b-local"
SMOKE_REASONER_SPEC = "qwen3vl-2b-local"
DEFAULT_MAX_TOKENS = 256
SMOKE_MAX_TOKENS = 64


# Named visual-resolution presets. Value = per-page pixel cap = tokens_per_page *
# 28 * 28 (Qwen packs a 28x28 patch per vision token). One preset is fixed as the
# study-wide deployment resolution (DEPLOYMENT_RESOLUTION); the scientific
# resolution sweep is the only thing that varies it. Resolution is the one
# representation parameter held identical across machines, since a lower-res image
# is a genuinely different (lossier) input.
VISUAL_RESOLUTION_PRESETS: dict[str, int] = {
    "full": 1280 * 28 * 28,  # 1,003,520 px  ~1280 tok/page
    "high": 768 * 28 * 28,   #   602,112 px   ~768 tok/page
    "med": 512 * 28 * 28,    #   401,408 px   ~512 tok/page
    "low": 320 * 28 * 28,    #   250,880 px   ~320 tok/page
    "min": 224 * 28 * 28,    #   175,616 px   ~224 tok/page
}

# The single fixed resolution used by every table except the scientific sweep.
# PLACEHOLDER: set to "med" so the pipeline has a concrete preset to run at. This
# is NOT the final value. The operational resolution probe (job 1017226, V rung,
# worst-case ~10 pages, 16GB V100) reports the highest preset that fits; that
# verdict replaces this. Re-check if the parser path shifts the sequence profile.
DEPLOYMENT_RESOLUTION = "med"


def max_pixels_for_resolution(config: "ExperimentConfig") -> int:
    """Per-page pixel cap for a run's chosen visual-resolution preset."""

    return VISUAL_RESOLUTION_PRESETS[config.visual_resolution]


@dataclass(frozen=True)
class ExperimentConfig:
    """The knobs one experiment run reads."""

    dataset: str = "mmlongbench"
    smoke: bool = False

    # Reasoner: 8B is the center config; the size sweep runs 2B/4B/8B/32B. Specs
    # are 'family-size-backend' strings.
    reasoner_spec: str = DEFAULT_REASONER_SPEC
    scaling_specs: tuple[str, ...] = (
        "qwen3vl-2b-local",
        "qwen3vl-4b-local",
        "qwen3vl-8b-local",
        "qwen3vl-32b-local",
    )
    judge_spec: str = "stub"

    # Input conditions and the top-k depths swept for retrieved conditions.
    conditions: tuple[str, ...] = ("oracle", "retrieved", "full", "similarity")
    k_values: tuple[int, ...] = (1, 3, 5, 7, 10)

    # Representation ladder (cost-ordered; names historical, mechanism in tools/).
    representations: tuple[str, ...] = ("T", "TL", "TLV", "V")

    bins: tuple[str, ...] = DEFAULT_BINS
    cost_metric: str = "latency_bs1"

    # Pre-registered sufficiency margin in accuracy points.
    sufficiency_margin: float = 3.0

    # Rendering / sampling knobs.
    dpi: int = 144
    sample: int | None = None
    # Per-bin document-level subset for full runs: whole documents are drawn per
    # bin until it reaches this many questions, preserving doc-coherent sampling
    # for the doc-level bootstrap CIs. Set to 0/None to run the whole corpus.
    per_bin_sample: int | None = 100
    sample_seed: int = 0
    # Optional bitsandbytes quantization for the local reasoner: None (bf16),
    # "4bit", or "8bit". When set it is appended to `reasoner_spec` as a
    # `-4bit`/`-8bit` suffix so the quantized run gets its own cache rows.
    quantization: str | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS

    # The single visual-resolution preset this run feeds every reasoner. Defaults
    # to the fixed deployment resolution; the scientific resolution sweep is the
    # only run that overrides it per table.
    visual_resolution: str = DEPLOYMENT_RESOLUTION

    # Optional per-run cache namespace nested under the versioned cache root, so
    # two runs sharing an experiment selection (e.g. two full runs with different
    # reasoners) never write the same files. Judge/build must pass the same tag.
    run_tag: str | None = None

    paths: ProjectPaths = field(default_factory=ProjectPaths)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bins", tuple(self.bins))
        object.__setattr__(self, "representations", tuple(self.representations))
        if self.smoke:
            object.__setattr__(self, "reasoner_spec", SMOKE_REASONER_SPEC)
            object.__setattr__(self, "max_tokens", min(int(self.max_tokens), SMOKE_MAX_TOKENS))
        if self.quantization is not None:
            if self.quantization not in ("4bit", "8bit"):
                raise ValueError(f"quantization must be None, '4bit', or '8bit', got {self.quantization!r}")
            object.__setattr__(self, "reasoner_spec", f"{self.reasoner_spec}-{self.quantization}")
        if self.visual_resolution not in VISUAL_RESOLUTION_PRESETS:
            raise ValueError(
                f"visual_resolution must be one of {sorted(VISUAL_RESOLUTION_PRESETS)}, "
                f"got {self.visual_resolution!r}"
            )
        if self.run_tag is not None:
            tag = self.run_tag
            if not tag or not all(ch.isalnum() or ch in "-_" for ch in tag):
                raise ValueError(f"run_tag must be non-empty alphanumeric/dash/underscore, got {tag!r}")
            object.__setattr__(self, "paths", replace(self.paths, cache_dir=self.paths.cache_dir / tag))
