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

@dataclass(frozen=True)
class ProjectPaths:
    """Root-relative artifact paths shared by local and Kaya execution."""

    root: Path = ROOT
    data_dir: Path = ROOT / ".data"
    hf_home: Path = ROOT / ".cache"
    results_dir: Path = ROOT / "results"
    # Cached cells live under `results/cache/<run_tag>/…`; a run_tag isolates one
    # run's cells so an unrelated run can never read them back.
    cache_dir: Path = ROOT / "results" / "cache"
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

# Instruction preambles for the abstention/hallucination prompt sweep. The mode
# rides on the cell's conditioner name (like the retrieval k does), so each mode
# is its own cached cell, and the reasoner applies the matching instruction. The
# targeted text is the one every answerable (G1/G2) cell uses.
PROMPT_MODES: dict[str, str] = {
    "none": "",
    "generic": "Use only the provided document evidence and keep the answer concise.",
    "targeted": (
        "Use only the provided document evidence. If the evidence does not contain the answer, "
        "answer exactly: Not answerable.\nKeep the answer concise."
    ),
}
DEFAULT_PROMPT_MODE = "targeted"
# The conditions the hallucination task sweeps: no guidance, generic, and
# abstention-targeted. The prompting comparison needs the unprompted (none) arm as
# its baseline, so all three ride their own cache cells.
G3_PROMPT_MODES: tuple[str, ...] = ("none", "generic", "targeted")


# Named visual-resolution presets. Value = per-page pixel cap = tokens_per_page *
# 28 * 28 (Qwen packs a 28x28 patch per vision token). One preset is fixed as the
# study-wide deployment resolution (DEPLOYMENT_RESOLUTION); the scientific
# resolution sweep is the only thing that varies it. Resolution is the one
# representation parameter held identical across machines, since a lower-res image
# is a genuinely different (lossier) input.
VISUAL_RESOLUTION_PRESETS: dict[str, int] = {
    "high": 960 * 28 * 28,   #   752,640 px   ~960 tok/page
    "med": 640 * 28 * 28,    #   501,760 px   ~640 tok/page
    "low": 400 * 28 * 28,    #   313,600 px   ~400 tok/page
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


def hf_cache_environ(cache_dir: Path) -> dict[str, str]:
    """Cache-location env vars that point every model download and load, plus the
    parser subprocesses, at `cache_dir`.

    Shared by prestage (which forces them before downloading) and the generate
    entry point (which sets them only if unset, so a Kaya run's own exports win).
    This is what lets a direct run on another machine, e.g. the H100 supervisor,
    find the prestaged weights without any manual exports.
    """

    cache = str(cache_dir)
    return {
        "HF_HOME": cache,
        "HF_HUB_CACHE": cache,
        "HF_XET_CACHE": str(Path(cache) / "xet"),
        "MODELSCOPE_CACHE": str(Path(cache) / "modelscope"),
        "MINERU_MODEL_SOURCE": "huggingface",
        "PADDLE_PDX_MODEL_SOURCE": "huggingface",
    }


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
    # Optional explicit reasoner list for a size/family sweep. When non-empty the
    # reasoner tasks generate one pass per spec (freeing the GPU between them)
    # instead of the single reasoner_spec; empty means just reasoner_spec. Set it
    # to `scaling_specs` for the model-size sweep.
    reasoner_specs: tuple[str, ...] = ()
    judge_spec: str = "stub"

    # Input conditions and the top-k depths swept for retrieved conditions.
    conditions: tuple[str, ...] = ("oracle", "retrieved", "full", "similarity")
    k_values: tuple[int, ...] = (1, 3, 5, 7, 10)

    # Representation ladder (cost-ordered; names historical, mechanism in tools/).
    representations: tuple[str, ...] = ("T", "TL", "TLV", "V")

    # PDF parser feeding the TL/TLV text channel. The parser comparison varies
    # this per run (as its own run_tag); T and V never use it.
    parser_tool: str = "paddleocrvl"

    bins: tuple[str, ...] = DEFAULT_BINS
    cost_metric: str = "latency_bs1"

    # Pre-registered sufficiency margin in accuracy points.
    sufficiency_margin: float = 3.0

    # Rendering / sampling knobs. dpi is the render resolution the OCR/parser sees
    # (the VLM downsamples to visual_resolution), so it is set for parser quality.
    dpi: int = 200
    sample: int | None = None
    # Per-bin document-level subset for full runs: whole documents are drawn per
    # bin until it reaches this many questions, preserving doc-coherent sampling
    # for the doc-level bootstrap CIs. Set to 0/None to run the whole corpus.
    per_bin_sample: int | None = 100
    sample_seed: int = 0
    # Per-doc_type document-level subset: whole documents are drawn per native
    # doc_type label until it reaches this many questions (same doc-coherent draw
    # as per_bin_sample). Set via a spec's {sampling: {per_doc_type: N, seed: S}};
    # None runs the whole pool.
    per_doc_type_sample: int | None = None
    # Optional bitsandbytes quantization for the local reasoner: None (bf16),
    # "4bit", or "8bit". When set it is appended to `reasoner_spec` as a
    # `-4bit`/`-8bit` suffix so the quantized run gets its own cache rows.
    quantization: str | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS

    # The visual-resolution preset this run feeds a cell when it is not sweeping.
    visual_resolution: str = DEPLOYMENT_RESOLUTION
    # Optional list of presets to sweep: the reasoner runs every cell once per
    # preset, and the preset is part of the cell key so the runs never collide.
    # Empty means just `visual_resolution`. Set it for the resolution sweep.
    visual_resolutions: tuple[str, ...] = ()

    # Optional per-run cache namespace nested under the versioned cache root, so
    # two runs sharing an experiment selection (e.g. two full runs with different
    # reasoners) never write the same files. Judge/build must pass the same tag.
    run_tag: str | None = None

    paths: ProjectPaths = field(default_factory=ProjectPaths)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bins", tuple(self.bins))
        object.__setattr__(self, "representations", tuple(self.representations))
        object.__setattr__(self, "reasoner_specs", tuple(self.reasoner_specs))
        if self.smoke:
            object.__setattr__(self, "reasoner_spec", SMOKE_REASONER_SPEC)
            object.__setattr__(self, "reasoner_specs", ())  # smoke never sweeps
            object.__setattr__(self, "max_tokens", min(int(self.max_tokens), SMOKE_MAX_TOKENS))
        if self.quantization is not None:
            if self.quantization not in ("4bit", "8bit"):
                raise ValueError(f"quantization must be None, '4bit', or '8bit', got {self.quantization!r}")
            object.__setattr__(self, "reasoner_spec", f"{self.reasoner_spec}-{self.quantization}")
            if self.reasoner_specs:
                object.__setattr__(
                    self, "reasoner_specs",
                    tuple(f"{spec}-{self.quantization}" for spec in self.reasoner_specs),
                )
        if self.parser_tool not in ("paddleocrvl", "mineru", "unlimited"):
            raise ValueError(
                f"parser_tool must be one of paddleocrvl/mineru/unlimited, got {self.parser_tool!r}"
            )
        if self.visual_resolution not in VISUAL_RESOLUTION_PRESETS:
            raise ValueError(
                f"visual_resolution must be one of {sorted(VISUAL_RESOLUTION_PRESETS)}, "
                f"got {self.visual_resolution!r}"
            )
        object.__setattr__(self, "visual_resolutions", tuple(self.visual_resolutions))
        unknown_res = [r for r in self.visual_resolutions if r not in VISUAL_RESOLUTION_PRESETS]
        if unknown_res:
            raise ValueError(
                f"visual_resolutions must each be one of {sorted(VISUAL_RESOLUTION_PRESETS)}, "
                f"got {unknown_res}"
            )
        if self.run_tag is not None:
            tag = self.run_tag
            if not tag or not all(ch.isalnum() or ch in "-_" for ch in tag):
                raise ValueError(f"run_tag must be non-empty alphanumeric/dash/underscore, got {tag!r}")
            object.__setattr__(self, "paths", replace(self.paths, cache_dir=self.paths.cache_dir / tag))
