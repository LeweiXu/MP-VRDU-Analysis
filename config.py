"""Run knobs shared across the pipeline: filesystem paths, model specs, named
visual-resolution presets, sampling defaults, and the scoring/evaluation constants
(bootstrap CI, abstention forms, judge rubric and models, representation ladder)."""

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
# native doc_type. Single source of truth: data.annotations.BIN_LABELS and
# data.binning.BINS import this.
DEFAULT_BINS: tuple[str, ...] = ("text-dominant", "mixed-modality", "visual-dominant")

# The cost-ordered representation ladder (rungs the reasoner climbs). Single source
# of truth: scoring.frontier.RUNG_ORDER and pipeline.representation.RUNGS import this,
# and it is the default for ExperimentConfig.representations.
REPRESENTATION_LADDER: tuple[str, ...] = ("T", "TL", "TLV", "V")

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

# The single baseline configuration each run is measured against. The experiment
# is one pipeline run at this baseline, and every sweep changes exactly one axis
# off it while holding the rest here fixed. The table build reads this as the
# source of truth for the held-fixed values it prints in each table's caption, so
# every result is explainable on its own. Per task; a swept axis overrides its
# entry for that table only.
BASELINE: dict[str, dict[str, str]] = {
    "G1_oracle_ladder": {
        "dataset": "mmlongbench",
        "scan": "any",
        "sampling": "full",
        "parser": "paddleocrvl",
        "reasoner_spec": "qwen3vl-8b-local",
        "quantization": "bf16",
        "visual_resolution": "med",
        "representation": "T/TL/TLV/V",
        "pool": "answerable",
        "page_selection": "oracle",
        "prompt_mode": "none",
    },
    "G2_retrieval": {
        "dataset": "mmlongbench",
        "scan": "any",
        "sampling": "full",
        "parser": "paddleocrvl",
        "reasoner_spec": "qwen3vl-8b-local",
        "quantization": "bf16",
        "visual_resolution": "med",
        "representation": "TLV/V",
        "pool": "answerable",
        "page_selection": "retrieved (bm25 text / colqwen2.5 vision / joint)",
        "prompt_mode": "none",
    },
    "G3_hallucination": {
        "dataset": "mmlongbench",
        "scan": "any",
        "sampling": "full",
        "parser": "paddleocrvl",
        "reasoner_spec": "qwen3vl-8b-local",
        "quantization": "bf16",
        "visual_resolution": "med",
        "representation": "TLV",
        "pool": "unanswerable",
        "page_selection": "similarity (bm25, k=3)",
        "prompt_mode": "none",
    },
}


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

# Qwen3-Embedding-4B (the expensive dense text retriever) memory knobs for a 16 GB
# V100. There is no FlashAttention on this card, so attention is O(seq^2): a long page
# must be capped or it OOMs on its own forward pass, and batch=1 bounds the batch
# dimension. 4096 fits with headroom and truncates only the rare very long page. Applied
# in retrievers/text.py; raise the cap (or drop batching) only on a larger GPU.
QWEN3_EMBEDDING_MAX_SEQ_LEN = 4096
QWEN3_EMBEDDING_ENCODE_BATCH = 1


# -- Scoring / evaluation constants (the science params) ---------------------
# These define how results are measured; centralised here so a run's evaluation
# is visible in one place rather than buried in scoring/ and pipeline/.

# Document-level bootstrap CI (scoring.accuracy): number of resamples, RNG seed,
# and the two-sided quantiles (2.5% / 97.5% = a 95% interval).
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 0
BOOTSTRAP_CI_LOW = 0.025
BOOTSTRAP_CI_HIGH = 0.975

# Normalised refusal / no-evidence surface forms an answer counts as abstention
# (scoring.abstention). Matched as substrings against the casefolded answer.
ABSTENTION_FORMS: tuple[str, ...] = (
    "not answerable",
    "cannot be answered",
    "can not be answered",
    "cannot answer",
    "unanswerable",
    "insufficient information",
    "not enough information",
    "no answer",
    "unknown from the document",
    "not mentioned",
    "not provided",
)

# A page is "text" if it has at least this many extracted characters; a document
# with too few is auto-labelled scanned (data.render). This is the digital/scanned
# corpus split threshold.
SCANNED_MIN_CHARS_PER_PAGE = 20

# LLM judge (pipeline.judge): the shared rubric and the two judge model ids. The
# judge model and prompt *are* the evaluation, so they live here.
JUDGE_GPT_MODEL = "gpt-4o-mini"
JUDGE_GEMINI_MODEL = "gemini-2.5-flash"
JUDGE_SYSTEM_PROMPT = """You judge answers to document questions.
Return only JSON with keys:
- verdict: one of correct, incorrect, abstained
- extracted_answer: the answer extracted from the model response, or empty string
- rationale: a short reason

Mark correct when the model answer is semantically equivalent to the gold answer.
For unanswerable questions, mark correct only when the model abstains.
"""


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

    # The document-classifier model for predicted-domain routing, priced once as a
    # side-artifact by G3. None (or "none") skips it, so routing reports only the
    # gold-bin ceiling. The classifier itself is a small first-N-pages pass.
    classifier_spec: str | None = None

    # Document scan filter applied BEFORE the pool + sampling (spec `corpus.scan`):
    # "any" (no filter), "digital", or "scanned". When set, documents are labelled by
    # PyMuPDF auto-detection, cached to annotations/auto_scan.csv.
    scan_filter: str = "any"
    # The question pool this run draws from (spec `corpus.pool`): "answerable",
    # "unanswerable", or "all" (both).
    pool: str = "answerable"
    # The corpus `sampling` block (spec `corpus.sampling`), applied after scan + pool:
    # "full", {per_doc_type: N, seed}, {per_bin: N, seed}, {limit: N}, or {ids: [...]}.
    sampling: object = "full"
    # How pages are selected for the reasoner, in the T/TL/TLV/V vocabulary the
    # retriever ranks over: ("oracle",) = gold pages; ("T",)/("V",)/("T","V") =
    # text (PyMuPDF) / vision (image) retrieval arms.
    retrieval_representation: tuple[str, ...] = ("oracle",)

    # Input conditions and the top-k depths swept for retrieved conditions.
    conditions: tuple[str, ...] = ("oracle", "retrieved", "full", "similarity")
    k_values: tuple[int, ...] = (1, 3, 5, 7, 10)

    # G2 retrieval benchmark: the text/vision methods scored into the side-artifact
    # (cost-ordered cheap -> expensive) and the joint unions. `joints` is "matched"
    # (auto cheap|cheap, mid|mid, expensive|expensive by list position) or an explicit
    # tuple of (text, vision) pairs, or () to skip joints. These lists are consumed
    # within one run (like k_values), so a failed method just skips its own rows.
    text_retrievers: tuple[str, ...] = ("bm25", "bge-m3", "qwen3-embedding")
    vision_retrievers: tuple[str, ...] = ("colmodernvbert", "colqwen2.5", "colqwen3")
    joints: object = "matched"
    joint_k_values: tuple[int, ...] = (1, 3, 5)

    # G2 inference stage: which single retriever arm feeds the reasoner (a subset of
    # the benchmark lists), whether to also feed the joint union, and at which rungs.
    inference_text_retriever: str = "bm25"
    inference_vision_retriever: str = "colqwen2.5"
    inference_joint: bool = True
    inference_representations: tuple[str, ...] = ("TLV", "V")

    # G3 abstention-prompt sweep: the instruction preambles run as separate cells in
    # one run. Defaults to the three-mode comparison.
    prompt_modes: tuple[str, ...] = G3_PROMPT_MODES

    # Representation ladder (cost-ordered; names historical, mechanism in tools/).
    representations: tuple[str, ...] = REPRESENTATION_LADDER

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
    # Per-doc_type subset: whole documents are drawn per native doc_type label, then
    # capped to EXACTLY this many questions per label (so per_doc_type: 1 -> one
    # question per label). The exact cap can slice the last drawn document. Set via a
    # spec's {sampling: {per_doc_type: N, seed: S}}; None runs the whole pool.
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
        if isinstance(self.classifier_spec, str) and self.classifier_spec.strip().lower() in ("", "none"):
            object.__setattr__(self, "classifier_spec", None)
        object.__setattr__(self, "representations", tuple(self.representations))
        object.__setattr__(self, "reasoner_specs", tuple(self.reasoner_specs))
        for name in ("text_retrievers", "vision_retrievers", "joint_k_values",
                     "inference_representations", "prompt_modes", "retrieval_representation"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if isinstance(self.joints, list):
            object.__setattr__(self, "joints", tuple(tuple(pair) for pair in self.joints))
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
