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


PLAN: tuple[AnalysisTable, ...] = (
    AnalysisTable("headline", "G1_oracle_ladder", (G1_BASE,),
                  "representation ladder T/TL/TLV/V", "headline.build", sweeps_key="representation",
                  summary="headline.summary"),
    AnalysisTable("parser", "G1_oracle_ladder", (G1_BASE, G1_PARSER_MINERU, G1_PARSER_UNLIMITED),
                  "parser (paddleocrvl / mineru / unlimited)", "parser.build", sweeps_key="parser",
                  parser_by_tag={G1_BASE: "paddleocrvl", G1_PARSER_MINERU: "mineru", G1_PARSER_UNLIMITED: "unlimited"},
                  caption_extra={"representation": "TL/TLV only"}, summary="parser.summary"),
    AnalysisTable("resolution", "G1_oracle_ladder", G1_RES,
                  "visual_resolution (low / med / high)", "resolution.build", sweeps_key="visual_resolution",
                  caption_extra={"scan": "any (digital+scanned)", "representation": "TLV/V only"},
                  summary="resolution.summary"),
    AnalysisTable("scale", "G1_oracle_ladder", (G1_BASE, *G1_SIZE),
                  "reasoner_spec (size + family)", "scale.build", sweeps_key="reasoner_spec",
                  caption_extra={"scan": "any (digital+scanned)", "note": "8b bf16 baseline from the representation run"}),
    AnalysisTable("quantization", "G1_oracle_ladder", (G1_BASE, *G1_QUANT),
                  "quantization (bf16 / 8bit / 4bit)", "mined_quant_sensitivity.build", sweeps_key="quantization",
                  caption_extra={"scan": "any (digital+scanned)", "note": "16-bit baseline from the representation run"},
                  summary="mined_quant_sensitivity.summary"),
    AnalysisTable("scan_vs_digital", "G1_oracle_ladder", (G1_BASE,),
                  "scan (digital vs scanned), all rungs", "mined_scan_vs_digital.build", sweeps_key="scan",
                  caption_extra={"scan": "split: digital vs scanned"}, summary="mined_scan_vs_digital.summary"),
    AnalysisTable("composition", "G1_oracle_ladder", (G1_BASE,),
                  "evidence source × rung", "composition.build", sweeps_key="representation"),
    AnalysisTable("routing", "G1_oracle_ladder", (G1_BASE,),
                  "routing policy", "routing.build",
                  caption_extra={"note": "assembled from G1 ladder rows + G3 classifier price"}),
    AnalysisTable("prefill_cost", "G1_oracle_ladder", (G1_BASE,),
                  "representation (prefill / input tokens)", "mined_prefill_cost.build", sweeps_key="representation",
                  summary="mined_prefill_cost.summary"),
    AnalysisTable("vram_headroom", "G1_oracle_ladder", G1_ALL,
                  "spec / rung / resolution (peak VRAM)", "mined_vram_headroom.build",
                  caption_extra={"note": "pooled across all G1 runs"}, summary="mined_vram_headroom.summary"),
    AnalysisTable("oom_frontier", "G1_oracle_ladder", G1_ALL,
                  "rung / resolution / pages-fed (OOM rate)", "mined_oom_frontier.build", reads="predictions",
                  caption_extra={"note": "OOM from status rows, pooled across all G1 runs"},
                  summary="mined_oom_frontier.summary"),
    AnalysisTable("matched_cross", "G2_retrieval", (G2,),
                  "retrieval modality (matched vs cross)", "matched_cross.build", sweeps_key="page_selection"),
    AnalysisTable("kdepth", "G2_retrieval", (G2,),
                  "retrieval depth k", "kdepth.build", sweeps_key="page_selection"),
    AnalysisTable("retrieval_accuracy", "G2_retrieval", (G2,),
                  "retriever × k (page P/R/F1)", "retrieval_accuracy.build", reads="side:retrieval.jsonl",
                  summary="retrieval_accuracy.summary", detail_md=False),
    AnalysisTable("retrieval_accuracy_overall", "G2_retrieval", (G2,),
                  "retriever × k (overall P/R/F1)", "retrieval_accuracy.build_overall", reads="side:retrieval.jsonl"),
    AnalysisTable("retrieval_dpi", "G2_retrieval", (G2,),
                  "render dpi × retriever × k", "retrieval_accuracy.build_by_dpi", reads="side:retrieval.jsonl"),
    AnalysisTable("hallucination", "G3_hallucination", (G3,),
                  "prompt_mode (none / generic / targeted)", "hallucination.build", sweeps_key="prompt_mode"),
    AnalysisTable("abstention_by_doctype", "G3_hallucination", (G3,),
                  "prompt_mode × doc_type", "mined_abstention_by_doctype.build", sweeps_key="prompt_mode"),
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
