"""The build plan: which analysis table reads which run_tag(s), the variable it
sweeps, and the caption pinning the held-fixed baseline it is measured against.

One entry per table. `caption_for` reads the swept axis plus `config.BASELINE` (the
held-fixed values) so every table is explainable on its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from config import BASELINE

G1_BASE = "g1-representation-full"
G1_PARSER_MINERU = "g1-parser-full-mineru"
G1_PARSER_UNLIMITED = "g1-parser-full-unlimited"
# G1 sweep run_tags whose digital run merges with its scanned half at build time.
G1_SIZE = ("g1-reasoner-full", "g1-reasoner-scanned")
G1_RES = ("g1-resolution-full", "g1-resolution-scanned")
G1_QUANT = ("g1-quantization-full", "g1-quantization-scanned")
# Every G1 run_tag, for the pooled telemetry (VRAM headroom, OOM frontier).
G1_ALL = (G1_BASE, *G1_SIZE, *G1_RES, *G1_QUANT)
G2 = "g2-retrieval-full"
G3 = "g3-hallucination-full"

# The report groups tables by the research question they answer. A table that maps
# to no RQ is retained under the appendix rather than dropped, which is why that is
# the default. Order here is the order the sections appear in all_tables.md.
RQ1, RQ2, RQ3 = "RQ1", "RQ2", "RQ3"
APPENDIX = "Appendix"
RQ_SECTIONS: tuple[tuple[str, str], ...] = (
    (RQ1, "RQ1 — Error attribution (where the loss is)"),
    (RQ2, "RQ2 — Deployment feasibility (what can be run)"),
    (RQ3, "RQ3 — Recoverable loss (which levers help)"),
    (APPENDIX, "Appendix — not mapped to an RQ, retained"),
)

# Tables built wholly or partly on the g2-retrieval-full generation pool, which was
# only ~36% pulled before the cluster migration with judging still in flight. Their
# captions carry this so a reader never mistakes them for settled numbers.
PROVISIONAL_G2 = {"status": "PROVISIONAL (partial G2 pool)"}

# Two places where a written summary and the emitted rows disagree. Both are surfaced
# on the affected captions rather than silently resolved in favour of one side.
G2_ARM_NOTE = (
    "spec ran bge-m3 text / colqwen2.5 vision / joint; BASELINE captions the text arm "
    "as bm25 (the config default the spec overrode) — cite the spec"
)
G3_SELECTION_NOTE = (
    "described as similarity (bm25, k=3); rows are emitted under base retrieved_text_k3 "
    "with provenance=retrieved"
)


@dataclass(frozen=True)
class AnalysisTable:
    """One output table: its source run_tag(s), the swept axis, and the builder."""

    key: str
    task: str
    run_tags: tuple[str, ...]
    swept: str
    builder: str                       # "<module>.<fn>" in reporting.tables
    reads: str = "results"             # results | predictions | side:<name>
    sweeps_key: str = ""               # BASELINE axis this table varies (dropped from held-fixed)
    parser_by_tag: Mapping[str, str] = field(default_factory=dict)
    caption_extra: Mapping[str, str] = field(default_factory=dict)
    summary: str = ""                  # "<module>.<fn>" for a doc_type-collapsed markdown-only summary
    detail_md: bool = True             # include the full detail table in all_tables.md
    rq: str = APPENDIX                 # report section this table lands under (see RQ_SECTIONS)


# Entries are grouped by the RQ section they land under and ordered within it; the
# report renders the sections in RQ_SECTIONS order, keeping this order inside each.
PLAN: tuple[AnalysisTable, ...] = (
    # -- RQ1: where the loss is introduced -----------------------------------
    AnalysisTable("headline", "G1_oracle_ladder", (G1_BASE,),
                  "representation ladder T/TL/TLV/V", "headline.build", sweeps_key="representation",
                  summary="headline.summary", rq=RQ1),
    AnalysisTable("fidelity_transition", "G1_oracle_ladder", (G1_BASE,),
                  "rung transition (TL→TLV, T→TL, T→TLV) × evidence source", "fidelity_transition.build",
                  sweeps_key="representation",
                  caption_extra={"pairing": "within-question, both rungs status==ok",
                                 "transition columns": "percentages summing to 100 per row "
                                                       "(parenthesised figure is the raw count)",
                                 "All sources row": "pooled over questions, each counted once; "
                                                    "not a column sum, has its own paired n"},
                  rq=RQ1),
    AnalysisTable("fidelity_transition_doctype", "G1_oracle_ladder", (G1_BASE,),
                  "rung transition (TL→TLV, T→TL, T→TLV) × doc_type", "fidelity_transition.build_by_doctype",
                  sweeps_key="representation",
                  caption_extra={"pairing": "within-question, both rungs status==ok",
                                 "transition columns": "percentages summing to 100 per row "
                                                       "(parenthesised figure is the raw count)"},
                  summary="fidelity_transition.doctype_summary", rq=RQ1),
    AnalysisTable("composition", "G1_oracle_ladder", (G1_BASE,),
                  "evidence source × rung", "composition.build", sweeps_key="representation", rq=RQ1),
    AnalysisTable("source_stratification", "G1_oracle_ladder", (G1_BASE,),
                  "source_dataset stratum × rung", "source_stratification.build", sweeps_key="representation",
                  caption_extra={"strata": "loader dataset id only; no upstream QA provenance exists"},
                  rq=RQ1),
    AnalysisTable("attribution", "G1_oracle_ladder", (G1_BASE,),
                  "loss channel (representation / retrieval / reasoning)", "attribution.build",
                  caption_extra={"retrieval source": f"{G2} generation rows, loaded by the builder",
                                 **PROVISIONAL_G2}, rq=RQ1),
    AnalysisTable("parser", "G1_oracle_ladder", (G1_BASE, G1_PARSER_MINERU, G1_PARSER_UNLIMITED),
                  "parser (paddleocrvl / mineru / unlimited)", "parser.build", sweeps_key="parser",
                  parser_by_tag={G1_BASE: "paddleocrvl", G1_PARSER_MINERU: "mineru", G1_PARSER_UNLIMITED: "unlimited"},
                  caption_extra={"representation": "TL/TLV only"}, summary="parser.summary", rq=RQ1),
    AnalysisTable("integration", "G1_oracle_ladder", (G1_BASE,),
                  "evidence hop (single vs multi) × rung", "integration.build", sweeps_key="representation",
                  caption_extra={"hop": "single/multi only, hop=none dropped",
                                 "gap column": "M − S = multi-page minus single-page accuracy, in points "
                                               "(negative = multi-page is worse)"},
                  summary="integration.summary", rq=RQ1),
    AnalysisTable("hop_rung", "G1_oracle_ladder", (G1_BASE,),
                  "hop_bucket (1 / 2 / 3 / 4-5 / 6+) × rung", "hop_rung.build",
                  sweeps_key="representation",
                  caption_extra={"hop": "bucketed evidence-page count, zero-evidence questions dropped",
                                 "tail buckets": "4-5 and 6+ are small; included for trend, not precision"},
                  summary="hop_rung.summary", rq=RQ1),
    AnalysisTable("hop_doctype", "G1_oracle_ladder", (G1_BASE,),
                  "doc_type × rung × evidence-page bucket (1 / 2 / 3+)", "hop_doctype.build",
                  sweeps_key="representation",
                  caption_extra={"buckets": "gold evidence-page count from the corpus annotation, "
                                            "zero-evidence questions dropped; 3+ merges the "
                                            "detail table's 3 / 4-5 / 6+ tail"},
                  rq=RQ1),
    AnalysisTable("hallucination", "G3_hallucination", (G3,),
                  "prompt_mode (none / generic / targeted; legacy names of the six-mode set)",
                  "hallucination.build", sweeps_key="prompt_mode",
                  caption_extra={"page_selection note": G3_SELECTION_NOTE}, rq=RQ1),
    AnalysisTable("abstention_by_doctype", "G3_hallucination", (G3,),
                  "prompt_mode × doc_type", "mined_abstention_by_doctype.build", sweeps_key="prompt_mode",
                  caption_extra={"page_selection note": G3_SELECTION_NOTE}, rq=RQ1),

    # -- RQ2: which representations can actually be run -----------------------
    AnalysisTable("scale", "G1_oracle_ladder", (G1_BASE, *G1_SIZE),
                  "reasoner_spec (size + family)", "scale.build", sweeps_key="reasoner_spec",
                  caption_extra={"scan": "any (digital+scanned)", "note": "8b bf16 baseline from the representation run"},
                  rq=RQ2),
    AnalysisTable("quantization", "G1_oracle_ladder", (G1_BASE, *G1_QUANT),
                  "quantization (bf16 / 8bit / 4bit)", "mined_quant_sensitivity.build", sweeps_key="quantization",
                  caption_extra={"scan": "any (digital+scanned)", "note": "16-bit baseline from the representation run"},
                  summary="mined_quant_sensitivity.summary", rq=RQ2),
    AnalysisTable("vram_headroom", "G1_oracle_ladder", G1_ALL,
                  "spec / rung / resolution (peak VRAM)", "mined_vram_headroom.build",
                  caption_extra={"note": "pooled across all G1 runs"}, summary="mined_vram_headroom.summary",
                  rq=RQ2),
    AnalysisTable("oom_frontier", "G1_oracle_ladder", G1_ALL,
                  "rung / resolution / pages-fed (OOM rate)", "mined_oom_frontier.build", reads="predictions",
                  caption_extra={"note": "OOM from status rows, pooled across all G1 runs"},
                  summary="mined_oom_frontier.summary", rq=RQ2),
    AnalysisTable("prefill_cost", "G1_oracle_ladder", (G1_BASE,),
                  "representation (prefill / input tokens)", "mined_prefill_cost.build", sweeps_key="representation",
                  summary="mined_prefill_cost.summary", rq=RQ2),

    # -- RQ3: which located losses an inference-time lever can repair ---------
    AnalysisTable("resolution", "G1_oracle_ladder", G1_RES,
                  "visual_resolution (low / med / high)", "resolution.build", sweeps_key="visual_resolution",
                  caption_extra={"scan": "any (digital+scanned)", "representation": "TLV/V only"},
                  summary="resolution.summary", rq=RQ3),
    AnalysisTable("matched_cross", "G2_retrieval", (G2,),
                  "retrieval modality (matched vs cross)", "matched_cross.build", sweeps_key="page_selection",
                  caption_extra={"inference arms": G2_ARM_NOTE, **PROVISIONAL_G2}, rq=RQ3),
    AnalysisTable("kdepth", "G2_retrieval", (G2,),
                  "retrieval depth k", "kdepth.build", sweeps_key="page_selection",
                  caption_extra={"inference arms": G2_ARM_NOTE, **PROVISIONAL_G2}, rq=RQ3),
    AnalysisTable("routing", "G1_oracle_ladder", (G1_BASE,),
                  "routing policy", "routing.build",
                  caption_extra={"note": "assembled from G1 ladder rows + G3 classifier price"}, rq=RQ3),

    # -- Appendix: retained, not mapped to an RQ ------------------------------
    AnalysisTable("retrieval_accuracy", "G2_retrieval", (G2,),
                  "retriever × k (page P/R/F1)", "retrieval_accuracy.build", reads="side:retrieval.jsonl",
                  summary="retrieval_accuracy.summary", detail_md=False),
    AnalysisTable("retrieval_accuracy_overall", "G2_retrieval", (G2,),
                  "retriever × k (overall P/R/F1)", "retrieval_accuracy.build_overall", reads="side:retrieval.jsonl"),
    AnalysisTable("retrieval_dpi", "G2_retrieval", (G2,),
                  "render dpi × retriever × k", "retrieval_accuracy.build_by_dpi", reads="side:retrieval.jsonl"),
    AnalysisTable("scan_vs_digital", "G1_oracle_ladder", (G1_BASE,),
                  "scan (digital vs scanned), all rungs", "mined_scan_vs_digital.build", sweeps_key="scan",
                  caption_extra={"scan": "split: digital vs scanned"}, summary="mined_scan_vs_digital.summary"),
)


def caption_for(entry: AnalysisTable) -> dict[str, str]:
    """Structured caption: the swept axis, then the held-fixed baseline for this task
    (minus the swept axis), then any per-table overrides/notes."""

    caption = {"swept": entry.swept}
    held = dict(BASELINE.get(entry.task, {}))
    if entry.sweeps_key:
        held.pop(entry.sweeps_key, None)
    caption.update(held)
    caption.update(entry.caption_extra)
    return caption
