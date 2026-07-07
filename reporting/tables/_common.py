"""Shared helpers and constants for the per-table builders.

Purpose:
    The pieces every `T*_*.py` table builder reuses: the table key/filename/title
    maps, the row->bin helpers, evidence-source normalisation, and the per-rung
    accuracy/cost aggregation. Keeping them here lets each table module stay just
    its own builder.

Pipeline role:
    Leaf module. The table builders and the `reporting.tables` entry point import
    from here; it imports nothing from within the package.

Arguments:
    None. Import-only; each helper takes result rows or metric summaries.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from data.binning import DocTypeBin, doc_type_bin
from metrics.accuracy import AccuracySummary, accuracy_summary
from metrics.cost import CostSummary, cost_summary
from metrics.frontier import FrontierCell, RUNG_ORDER
from pipeline.orchestrator import ResultRow


# Table key -> CSV filename. The keys ("table1".."table8") are the canonical
# identifiers used across the builders, the reporting routing, and the markdown.
TABLE_FILENAMES: dict[str, str] = {
    "table1": "table1_headline.csv",
    "table2": "table2_analytical.csv",
    "table3": "table3_family_replication.csv",
    "table4": "table4_dataset_replication.csv",
    "table5": "table5_composition_mediation.csv",
    "table6": "table6_matched_vs_cross.csv",
    "table7": "table7_routing.csv",
    "table8": "table8_scale_sanity.csv",
}

# Human titles for the combined markdown report; keys line up with TABLE_FILENAMES.
TABLE_TITLES: dict[str, str] = {
    "table1": "Headline frontier (doc-type bins x representation ladder)",
    "table2": "Analytical breakdown by question type",
    "table3": "Family replication (reasoner families)",
    "table4": "Dataset replication (held-out subset)",
    "table5": "Composition and mediation by evidence modality",
    "table6": "Matched vs cross retrieval",
    "table7": "Routing policies",
    "table8": "Scale sanity (2B/4B/8B/32B)",
}

QUESTION_TYPES = ("single-hop text", "table", "chart-figure", "multi-hop")
ROUTING_POLICIES = (
    "oracle_routing",
    "predicted_routing",
    "uniform_cheapest_T",
    "uniform_strongest_TLV",
)


def load_result_rows(path: Path) -> list[ResultRow]:
    """Load cached result rows from a jsonl file."""

    if not path.exists():
        return []
    rows: list[ResultRow] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(ResultRow.from_dict(json.loads(line)))
    return rows


def _bin(row: ResultRow) -> DocTypeBin:
    """Return the Option-A bin for a result row."""

    return doc_type_bin(row.doc_type)


def _safe_bin(row: ResultRow) -> str:
    """Return the Option-A bin for MMLongBench rows, or empty for other datasets."""

    try:
        return doc_type_bin(row.doc_type)
    except Exception:
        return ""


def _normalise_source(source: str) -> str:
    """Map raw evidence-source labels to the five mechanism modalities."""

    text = str(source).strip().casefold()
    if "table" in text:
        return "table"
    if "chart" in text:
        return "chart"
    if "figure" in text or "image" in text or "picture" in text:
        return "figure"
    if "layout" in text or "bbox" in text or "box" in text:
        return "layout"
    if "text" in text or "plain" in text:
        return "text"
    return "text" if not text or text == "none" else text.replace(" ", "_")


def analytical_question_type(row: ResultRow) -> str:
    """Return the Table-2 analytical question-type bucket for a row."""

    if row.hop == "multi":
        return "multi-hop"
    sources = {_normalise_source(source) for source in row.evidence_sources}
    if sources.intersection({"chart", "figure"}):
        return "chart-figure"
    if sources.intersection({"table", "layout"}):
        return "table"
    return "single-hop text"


def _metric_columns(prefix: str, accuracy: AccuracySummary, cost: CostSummary) -> dict[str, object]:
    """Return standard metric columns for one cell prefix."""

    return {
        f"{prefix}_n": accuracy.n_rows,
        f"{prefix}_docs": accuracy.n_docs,
        f"{prefix}_acc": accuracy.accuracy,
        f"{prefix}_ci_low": accuracy.ci_low,
        f"{prefix}_ci_high": accuracy.ci_high,
        f"{prefix}_latency_bs1_s": cost.latency_bs1_s,
        f"{prefix}_text_tokens": cost.input_text_tokens,
        f"{prefix}_vision_tokens": cost.input_visual_tokens,
        f"{prefix}_output_tokens": cost.output_tokens,
    }


def _rung_metrics(
    rows: Sequence[ResultRow],
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[dict[str, object], dict[str, FrontierCell], dict[str, CostSummary]]:
    """Return per-rung metric columns, frontier cells, and cost summaries."""

    columns: dict[str, object] = {}
    frontier_cells: dict[str, FrontierCell] = {}
    costs: dict[str, CostSummary] = {}
    for rung in RUNG_ORDER:
        rung_rows = [row for row in rows if row.representation == rung]
        acc = accuracy_summary(rung_rows, n_bootstrap=n_bootstrap, seed=seed)
        cost = cost_summary(rung_rows)
        columns.update(_metric_columns(rung, acc, cost))
        costs[rung] = cost
        if rung_rows:
            frontier_cells[rung] = FrontierCell(acc.accuracy, acc.ci_high)
    return columns, frontier_cells, costs


def _unique_question_count(rows: Sequence[ResultRow]) -> int:
    """Return number of distinct questions in a row group."""

    return len({row.question_id for row in rows})


def _unique_doc_count(rows: Sequence[ResultRow]) -> int:
    """Return number of distinct documents in a row group."""

    return len({row.doc_id for row in rows})


def _selected_row_map(rows: Sequence[ResultRow]) -> dict[tuple[str, str], ResultRow]:
    """Return one row per (question, representation), preferring oracle rows."""

    selected: dict[tuple[str, str], ResultRow] = {}
    for row in rows:
        key = (row.question_id, row.representation)
        if key not in selected or row.condition == "oracle":
            selected[key] = row
    return selected


def _size_label(model_spec: str) -> str:
    """Extract a readable model size label from a model spec."""

    parts = model_spec.split("-")
    for part in parts:
        if part.endswith("b") and part[:-1].isdigit():
            return part
    return ""
