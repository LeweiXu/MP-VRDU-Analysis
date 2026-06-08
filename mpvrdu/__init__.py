"""MP-VRDU component-analysis harness.

Top-level package. Sub-packages map onto the three-stage pipeline:

    represent/  -> Stage 2: parsers / OCR -> text + chunking
    retrieve/   -> Stage 1: evidence selection (baselines + retrievers)
    generate/   -> Stage 3: VLM wrappers + image/text/both input builder
    eval/       -> metrics + judge
    data/       -> dataset loading, PDF rendering, dev-slice carving
    pipeline.py -> wires the stages per config, emits per-question JSONL

See ../context.md and ../agent_build_plan.md.
"""

# MUST run before any transformers/huggingface_hub import so model + dataset
# downloads land in the repo-local cache (or Kaya's /group via HF_HOME).
from . import env as _env  # noqa: F401,E402

__version__ = "0.0.1"

