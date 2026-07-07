"""Aggregate cached result rows into the paper's CSV table shapes.

Purpose:
    Implements the Stage-M5 reporting surface. The builders consume judged
    `ResultRow` objects and emit the eight table shapes required by the paper,
    filled with whatever smoke/full rows are currently cached.

Pipeline role:
    `experiments.reporting` calls these builders (routing each table's source-task
    rows to it) after `results/cache/` is filled. Keeping aggregation here lets
    tables be rebuilt without rerunning models, judges, parsers, or retrieval.

Arguments:
    None. This module is import-only; callers pass result rows to
    `build_all_tables()` or call `write_all_tables()`.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import pandas as pd

from config import DEFAULT_BINS
from data.binning import DocTypeBin, doc_type_bin
from metrics.accuracy import AccuracySummary, accuracy_summary
from metrics.cost import CostSummary, cost_summary
from metrics.frontier import FrontierCell, RUNG_ORDER, sufficiency_frontier
from pipeline.orchestrator import ResultRow


TABLE_FILENAMES: Mapping[str, str] = {
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
TABLE_TITLES: Mapping[str, str] = {
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


def _group(rows: Iterable[ResultRow], *attrs: str) -> dict[tuple[str, ...], list[ResultRow]]:
    """Group rows by string attributes."""

    groups: dict[tuple[str, ...], list[ResultRow]] = defaultdict(list)
    for row in rows:
        key = tuple(str(getattr(row, attr)) for attr in attrs)
        groups[key].append(row)
    return groups


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


def build_table1_headline(
    rows: Sequence[ResultRow],
    *,
    bins: Sequence[str] = DEFAULT_BINS,
    margin_points: float = 3.0,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Build Table 1: bin x ladder headline with frontier and latency."""

    out: list[dict[str, object]] = []
    for bin_name in bins:
        group_rows = [row for row in rows if _bin(row) == bin_name]
        columns, cells, costs = _rung_metrics(group_rows, n_bootstrap=n_bootstrap, seed=seed)
        frontier = sufficiency_frontier(cells, margin_points=margin_points)
        out.append(
            {
                "bin": bin_name,
                "n_questions": _unique_question_count(group_rows),
                "n_docs": _unique_doc_count(group_rows),
                **columns,
                "frontier": frontier,
                "latency_at_frontier_s": costs[frontier].latency_bs1_s if frontier else 0.0,
            }
        )
    return pd.DataFrame(out)


def build_table2_analytical(
    rows: Sequence[ResultRow],
    *,
    bins: Sequence[str] = DEFAULT_BINS,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Build Table 2: bin x analytical question-type slice."""

    out: list[dict[str, object]] = []
    for bin_name in bins:
        for question_type in QUESTION_TYPES:
            group_rows = [
                row
                for row in rows
                if _safe_bin(row) == bin_name and analytical_question_type(row) == question_type
            ]
            columns, _, _ = _rung_metrics(group_rows, n_bootstrap=n_bootstrap, seed=seed)
            out.append(
                {
                    "bin": bin_name,
                    "question_type": question_type,
                    "n_questions": _unique_question_count(group_rows),
                    "n_docs": _unique_doc_count(group_rows),
                    **columns,
                }
            )
    return pd.DataFrame(out)


def build_table3_family_replication(
    rows: Sequence[ResultRow],
    *,
    bins: Sequence[str] = DEFAULT_BINS,
    margin_points: float = 3.0,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Build Table 3: model-family replication with primary-frontier match."""

    out: list[dict[str, object]] = []
    model_specs = sorted({row.model_spec for row in rows if _safe_bin(row)}) or [""]
    primary_spec = next((spec for spec in model_specs if spec.startswith("qwen3vl-8b")), model_specs[0])
    primary_frontiers: dict[str, str] = {}
    for bin_name in bins:
        primary_rows = [row for row in rows if row.model_spec == primary_spec and _safe_bin(row) == bin_name]
        _, cells, _ = _rung_metrics(primary_rows, n_bootstrap=n_bootstrap, seed=seed)
        primary_frontiers[bin_name] = sufficiency_frontier(cells, margin_points=margin_points)
    for model_spec in model_specs:
        model_rows = [row for row in rows if row.model_spec == model_spec]
        for bin_name in bins:
            group_rows = [row for row in model_rows if _safe_bin(row) == bin_name]
            columns, cells, _ = _rung_metrics(group_rows, n_bootstrap=n_bootstrap, seed=seed)
            frontier = sufficiency_frontier(cells, margin_points=margin_points)
            out.append(
                {
                    "model_spec": model_spec,
                    "model_size": _size_label(model_spec),
                    "bin": bin_name,
                    "n_questions": _unique_question_count(group_rows),
                    **columns,
                    "frontier": frontier,
                    "primary_model_spec": primary_spec,
                    "primary_frontier": primary_frontiers.get(bin_name, ""),
                    "matches_primary_frontier": bool(frontier and frontier == primary_frontiers.get(bin_name, "")),
                }
            )
    return pd.DataFrame(out)


def build_table4_dataset_replication(
    rows: Sequence[ResultRow],
    *,
    bins: Sequence[str] = DEFAULT_BINS,
    margin_points: float = 3.0,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Build Table 4: RQ1 replication on the held-out MMLongBench subset, per domain.

    Rows come from a disjoint set of MMLongBench documents (text_heavy /
    in_between) plus the reused visual_heavy questions (see
    `experiments/corpus.py::sample_table4_replication`), so this bins by the same
    three Option-A domains as Table 1 and marks each bin's frontier. Compare the
    frontier column against Table 1 to judge whether the recipe replicates.
    """

    out: list[dict[str, object]] = []
    for bin_name in bins:
        group_rows = [row for row in rows if _safe_bin(row) == bin_name]
        columns, cells, costs = _rung_metrics(group_rows, n_bootstrap=n_bootstrap, seed=seed)
        frontier = sufficiency_frontier(cells, margin_points=margin_points)
        out.append(
            {
                "dataset": "mmlongbench_heldout",
                "bin": bin_name,
                "n_questions": _unique_question_count(group_rows),
                "n_docs": _unique_doc_count(group_rows),
                **columns,
                "frontier": frontier,
                "latency_at_frontier_s": costs[frontier].latency_bs1_s if frontier else 0.0,
            }
        )
    return pd.DataFrame(out)


def build_table5_composition_mediation(
    rows: Sequence[ResultRow],
    *,
    bins: Sequence[str] = DEFAULT_BINS,
    margin_points: float = 3.0,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Build Table 5: evidence-composition mediation by bin and modality."""

    modalities = ("text", "table", "chart", "figure", "layout")
    out: list[dict[str, object]] = []
    bin_frontiers: dict[str, str] = {}
    modality_frontiers: dict[str, str] = {}

    for bin_name in bins:
        bin_rows = [row for row in rows if _safe_bin(row) == bin_name]
        _, cells, _ = _rung_metrics(bin_rows, n_bootstrap=n_bootstrap, seed=seed)
        bin_frontiers[bin_name] = sufficiency_frontier(cells, margin_points=margin_points)

    for modality in modalities:
        modality_rows = [
            row
            for row in rows
            if modality in {_normalise_source(source) for source in row.evidence_sources}
        ]
        _, cells, _ = _rung_metrics(modality_rows, n_bootstrap=n_bootstrap, seed=seed)
        modality_frontiers[modality] = sufficiency_frontier(cells, margin_points=margin_points)

    for bin_name in bins:
        bin_rows = [row for row in rows if _safe_bin(row) == bin_name]
        question_sources: dict[str, set[str]] = {}
        for row in bin_rows:
            question_sources.setdefault(row.question_id, set()).update(
                _normalise_source(source) for source in (row.evidence_sources or ("text",))
            )
        contributions = {modality: 0.0 for modality in modalities}
        for sources in question_sources.values():
            known = tuple(source for source in sources if source in contributions) or ("text",)
            weight = 1.0 / len(known)
            for source in known:
                contributions[source] += weight
        total = sum(contributions.values()) or 1.0
        shares = {modality: contributions[modality] / total for modality in modalities}
        predicted = predict_frontier_from_composition(shares, modality_frontiers)
        for modality in modalities:
            group_rows = [
                row
                for row in bin_rows
                if modality in {_normalise_source(source) for source in (row.evidence_sources or ("text",))}
            ]
            columns, _, _ = _rung_metrics(group_rows, n_bootstrap=n_bootstrap, seed=seed)
            out.append(
                {
                    "bin": bin_name,
                    "evidence_modality": modality,
                    "share": shares[modality],
                    "n_questions": _unique_question_count(group_rows),
                    "n_docs": _unique_doc_count(group_rows),
                    "modality_frontier": modality_frontiers.get(modality, ""),
                    "bin_frontier": bin_frontiers.get(bin_name, ""),
                    "predicted_bin_frontier": predicted,
                    "predicted_matches_bin": bool(predicted and predicted == bin_frontiers.get(bin_name, "")),
                    **columns,
                }
            )
    return pd.DataFrame(out)


def predict_frontier_from_composition(
    shares: Mapping[str, float],
    modality_frontiers: Mapping[str, str],
    *,
    min_share: float = 0.10,
) -> str:
    """Predict a bin frontier from modality shares and per-modality frontiers."""

    candidates = [
        modality_frontiers[modality]
        for modality, share in shares.items()
        if share >= min_share and modality_frontiers.get(modality)
    ]
    if not candidates:
        candidates = [frontier for frontier in modality_frontiers.values() if frontier]
    if not candidates:
        return ""
    rank = {rung: index for index, rung in enumerate(RUNG_ORDER)}
    return max(candidates, key=lambda rung: rank.get(rung, -1))


def _condition_k(condition: str) -> int | None:
    """Parse the top-k from a `retrieved_{modality}_k{K}` conditioner name."""

    match = re.search(r"_k(\d+)$", str(condition))
    return int(match.group(1)) if match else None


def _retrieval_summary_for(
    records: Sequence[Mapping[str, object]],
    *,
    doc_ids: set[str],
    modality: str,
    k: int | None = None,
) -> dict[str, float]:
    """Return macro retrieval metrics for selected docs/modality (optionally one k)."""

    selected = [
        record
        for record in records
        if str(record.get("doc_id", "")) in doc_ids
        and str(record.get("modality", "")) == modality
        and (k is None or "k" not in record or int(record.get("k", -1)) == k)
    ]
    if not selected:
        return {"retrieval_precision": 0.0, "retrieval_recall": 0.0, "retrieval_f1": 0.0}
    n = len(selected)
    return {
        "retrieval_precision": sum(float(record.get("precision", 0.0)) for record in selected) / n,
        "retrieval_recall": sum(float(record.get("recall", 0.0)) for record in selected) / n,
        "retrieval_f1": sum(float(record.get("f1", 0.0)) for record in selected) / n,
    }


def build_table6_matched_vs_cross(
    rows: Sequence[ResultRow],
    *,
    bins: Sequence[str] = DEFAULT_BINS,
    margin_points: float = 3.0,
    retrieval_records: Sequence[Mapping[str, object]] | None = None,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Build Table 6: matched-vs-cross retrieval on bins where vision helps.

    A bin is included when reasoning *materially benefits* from vision: the best
    vision-bearing rung (TLV/V) beats the best text-only rung (T/TL) by at least
    the sufficiency margin. This is looser than "vision is the frontier" (which
    also requires vision to be the *cheapest* sufficient rung) - with wide per-bin
    CIs the frontier often lands on TL even where TLV is much higher, dropping
    bins where matched-vs-cross is still worth reporting.
    """

    oracle_rows = [row for row in rows if row.condition == "oracle" and _safe_bin(row)]
    oracle_table = build_table1_headline(
        oracle_rows,
        bins=bins,
        margin_points=margin_points,
        n_bootstrap=n_bootstrap,
        seed=seed,
    ).set_index("bin")
    margin = margin_points / 100.0
    vision_bins = {
        bin_name
        for bin_name in bins
        if bin_name in oracle_table.index
        and oracle_table.loc[bin_name, "n_questions"] > 0  # skip empty bins (0 - 0 >= 0)
        and max(oracle_table.loc[bin_name, "TLV_acc"], oracle_table.loc[bin_name, "V_acc"])
        - max(oracle_table.loc[bin_name, "T_acc"], oracle_table.loc[bin_name, "TL_acc"])
        >= margin
    }
    retrieval_records = list(retrieval_records or [])
    columns = [
        "bin",
        "k",
        "pipeline",
        "condition",
        "retrieval_modality",
        "reasoning_modality",
        "n_rows",
        "n_docs",
        "accuracy",
        "ci_low",
        "ci_high",
        "latency_bs1_s",
        "delta_accuracy_vs_matched",
        "delta_latency_vs_matched_s",
        "retrieval_precision",
        "retrieval_recall",
        "retrieval_f1",
    ]
    # The k-sweep produces retrieved_{modality}_k{K} conditions; report each k as
    # its own matched-vs-cross pair. Fall back to a single implicit k if none parse.
    sweep_ks = sorted(
        {
            _condition_k(row.condition)
            for row in rows
            if str(row.condition).startswith("retrieved_") and _condition_k(row.condition) is not None
        }
    )
    out: list[dict[str, object]] = []
    for bin_name in bins:
        if bin_name not in vision_bins:
            continue
        for k in sweep_ks:
            definitions = {
                "matched_vision": [
                    row
                    for row in rows
                    if _safe_bin(row) == bin_name
                    and row.condition == f"retrieved_vision_k{k}"
                    and row.representation in {"TLV", "V"}
                ],
                "cross_text_to_vision": [
                    row
                    for row in rows
                    if _safe_bin(row) == bin_name
                    and row.condition == f"retrieved_text_k{k}"
                    and row.representation in {"TLV", "V"}
                ],
            }
            baseline_acc = accuracy_summary(definitions["matched_vision"], n_bootstrap=n_bootstrap, seed=seed)
            baseline_cost = cost_summary(definitions["matched_vision"])
            for pipeline, group_rows in definitions.items():
                acc = accuracy_summary(group_rows, n_bootstrap=n_bootstrap, seed=seed)
                cost = cost_summary(group_rows)
                modality = "vision" if pipeline == "matched_vision" else "text"
                out.append(
                    {
                        "bin": bin_name,
                        "k": k,
                        "pipeline": pipeline,
                        "condition": ",".join(sorted({row.condition for row in group_rows})) if group_rows else "",
                        "retrieval_modality": modality,
                        "reasoning_modality": "vision",
                        "n_rows": acc.n_rows,
                        "n_docs": acc.n_docs,
                        "accuracy": acc.accuracy,
                        "ci_low": acc.ci_low,
                        "ci_high": acc.ci_high,
                        "latency_bs1_s": cost.latency_bs1_s,
                        "delta_accuracy_vs_matched": acc.accuracy - baseline_acc.accuracy,
                        "delta_latency_vs_matched_s": cost.latency_bs1_s - baseline_cost.latency_bs1_s,
                        **_retrieval_summary_for(
                            retrieval_records,
                            doc_ids={row.doc_id for row in group_rows},
                            modality=modality,
                            k=k,
                        ),
                    }
                )
    return pd.DataFrame(out, columns=columns)


def build_table7_routing(
    rows: Sequence[ResultRow],
    *,
    bins: Sequence[str] = DEFAULT_BINS,
    margin_points: float = 3.0,
    classifier_records: Sequence[Mapping[str, object]] | None = None,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Build Table 7: corpus-level routing policies with classifier cost."""

    oracle_rows = [row for row in rows if row.condition == "oracle" and _safe_bin(row)]
    table1 = build_table1_headline(
        oracle_rows,
        bins=bins,
        margin_points=margin_points,
        n_bootstrap=n_bootstrap,
        seed=seed,
    ).set_index("bin")
    recipes = {
        bin_name: str(table1.loc[bin_name, "frontier"])
        for bin_name in bins
        if bin_name in table1.index and str(table1.loc[bin_name, "frontier"])
    }
    row_map = _selected_row_map(oracle_rows)
    classifier_records = list(classifier_records or [])
    predicted_bins = {
        str(record.get("doc_id", "")): str(record.get("predicted_bin", ""))
        for record in classifier_records
    }
    total_classifier_latency = sum(float(record.get("latency_s", 0.0)) for record in classifier_records)

    def select_rows(policy: str) -> tuple[list[ResultRow], str]:
        selected: list[ResultRow] = []
        chosen: dict[str, int] = {}
        for base in oracle_rows:
            if base.representation != "T":
                continue
            gold_bin = _safe_bin(base)
            if policy == "oracle_routing":
                rung = recipes.get(gold_bin, "")
            elif policy == "predicted_routing":
                rung = recipes.get(predicted_bins.get(base.doc_id, gold_bin), "")
            elif policy == "uniform_cheapest_T":
                rung = "T"
            else:
                rung = "TLV"
            if not rung:
                continue
            chosen[rung] = chosen.get(rung, 0) + 1
            row = row_map.get((base.question_id, rung))
            if row is not None:
                selected.append(row)
        chosen_text = ";".join(f"{rung}:{count}" for rung, count in sorted(chosen.items()))
        return selected, chosen_text

    out: list[dict[str, object]] = []
    for policy in ROUTING_POLICIES:
        group_rows, chosen = select_rows(policy)
        acc = accuracy_summary(group_rows, n_bootstrap=n_bootstrap, seed=seed)
        cost = cost_summary(group_rows)
        classifier_latency = total_classifier_latency / acc.n_rows if policy == "predicted_routing" and acc.n_rows else 0.0
        out.append(
            {
                "policy": policy,
                "chosen_rungs": chosen,
                "n_rows": acc.n_rows,
                "n_docs": acc.n_docs,
                "accuracy": acc.accuracy,
                "ci_low": acc.ci_low,
                "ci_high": acc.ci_high,
                "latency_bs1_s": cost.latency_bs1_s,
                "classifier_latency_bs1_s": classifier_latency,
                "total_latency_bs1_s": cost.latency_bs1_s + classifier_latency,
                "text_tokens": cost.input_text_tokens,
                "vision_tokens": cost.input_visual_tokens,
                "classifier_docs": len(classifier_records) if policy == "predicted_routing" else 0,
            }
        )
    return pd.DataFrame(out)


def build_table8_scale_sanity(
    rows: Sequence[ResultRow],
    *,
    bins: Sequence[str] = DEFAULT_BINS,
    margin_points: float = 3.0,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Build Table 8: model-scale sanity skeleton."""

    table = build_table3_family_replication(
        rows,
        bins=bins,
        margin_points=margin_points,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    if "model_spec" in table:
        table.insert(0, "scale_family", table["model_spec"].map(lambda value: str(value).split("-", 1)[0]))
    return table


def build_all_tables(
    rows: Sequence[ResultRow],
    *,
    dataset: str = "mmlongbench",
    bins: Sequence[str] = DEFAULT_BINS,
    margin_points: float = 3.0,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Build all eight paper table shapes."""

    return {
        "table1": build_table1_headline(
            rows,
            bins=bins,
            margin_points=margin_points,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
        "table2": build_table2_analytical(rows, bins=bins, n_bootstrap=n_bootstrap, seed=seed),
        "table3": build_table3_family_replication(
            rows,
            bins=bins,
            margin_points=margin_points,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
        "table4": build_table4_dataset_replication(
            rows,
            bins=bins,
            margin_points=margin_points,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
        "table5": build_table5_composition_mediation(
            rows,
            bins=bins,
            margin_points=margin_points,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
        "table6": build_table6_matched_vs_cross(
            rows,
            bins=bins,
            margin_points=margin_points,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
        "table7": build_table7_routing(
            rows,
            bins=bins,
            margin_points=margin_points,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
        "table8": build_table8_scale_sanity(
            rows,
            bins=bins,
            margin_points=margin_points,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
    }


def _table_to_markdown(df: pd.DataFrame) -> str:
    """Render one table DataFrame as a GitHub markdown table.

    A non-empty table is rendered with rows (floats rounded for readability). An
    empty table still emits its column header plus one blank row, so the combined
    report shows the table's skeleton with blank fields instead of dropping it.
    """

    columns = [str(col) for col in df.columns]
    if not columns:
        return "_(no columns)_"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    if df.empty:
        blank = "| " + " | ".join("" for _ in columns) + " |"
        return "\n".join([header, separator, blank])
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].round(4)
    body = [
        "| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |"
        for row in formatted.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *body])


def render_tables_markdown(
    tables: Mapping[str, pd.DataFrame],
    *,
    source: str | None = None,
    n_rows: int | None = None,
) -> str:
    """Render all eight tables into a single markdown document.

    Tables with data are filled; tables with no matching rows keep their skeleton
    (column header + a blank row) so the report always shows all eight shapes.
    """

    from datetime import datetime, timezone

    lines: list[str] = ["# MP-VRDU results tables", ""]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    meta = [f"Generated {stamp}."]
    if source:
        meta.append(f"Source: `{source}`.")
    if n_rows is not None:
        meta.append(f"{n_rows} result rows.")
    lines.append(" ".join(meta))
    lines.append("")
    lines.append("Empty tables show a blank skeleton row: their experiment has no cached rows yet.")
    lines.append("")
    for key in TABLE_FILENAMES:
        number = key.removeprefix("table")
        df = tables.get(key, pd.DataFrame())
        lines.append(f"## Table {number} — {TABLE_TITLES.get(key, key)}")
        note = "no data yet" if df.empty else f"{len(df)} rows"
        lines.append(f"_CSV: `{TABLE_FILENAMES[key]}` ({note})_")
        lines.append("")
        lines.append(_table_to_markdown(df))
        lines.append("")
    return "\n".join(lines)


def _fmt_pct(value: object, digits: int = 1) -> str:
    """Format a 0-1 accuracy as a percentage string, or '' if missing/non-numeric."""

    try:
        return f"{float(value) * 100:.{digits}f}"
    except (TypeError, ValueError):
        return ""


def _acc_ci(record: Mapping[str, object], prefix: str) -> str:
    """'44.4 [35.0, 53.7]' for a rung prefix (T/TL/TLV/V), or '' if no data."""

    acc = _fmt_pct(record.get(f"{prefix}_acc"))
    if not acc or not record.get(f"{prefix}_n"):
        return ""
    return f"{acc} [{_fmt_pct(record.get(f'{prefix}_ci_low'))}, {_fmt_pct(record.get(f'{prefix}_ci_high'))}]"


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render a github-flavoured markdown table."""

    if not rows:
        return "_(no rows)_"
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join("" if cell is None else str(cell) for cell in row) + " |")
    return "\n".join(out)


def _paper_ladder(df: pd.DataFrame, id_cols: Sequence[str], id_headers: Sequence[str], *, extra: Sequence[str] = ()) -> str:
    """Compact rung table: id cols, n, T/TL/TLV/V (acc [CI], frontier bold), frontier + latency."""

    rungs = ("T", "TL", "TLV", "V")
    headers = [*id_headers, "n", *rungs, "Frontier"]
    has_latency = "latency_at_frontier_s" in df.columns
    if "frontier" in df.columns and has_latency:
        headers.append("Frontier lat (s)")
    headers.extend(extra)
    rows: list[list[object]] = []
    for _, r in df.iterrows():
        front = str(r.get("frontier", "")) if "frontier" in df.columns else ""
        cells = []
        for rung in rungs:
            cell = _acc_ci(r, rung)
            cells.append(f"**{cell}**" if cell and rung == front else cell)
        row: list[object] = [r[c] for c in id_cols] + [int(r.get("n_questions", 0) or 0), *cells, front]
        if "frontier" in df.columns and has_latency:
            row.append(f"{float(r.get('latency_at_frontier_s', 0) or 0):.2f}")
        row.extend(r.get(c, "") for c in extra)
        rows.append(row)
    return _md_table(headers, rows)


def _paper_table2(df: pd.DataFrame) -> str:
    rungs = ("T", "TL", "TLV", "V")
    headers = ["Bin", "Question type", "n", *rungs]
    rows = [
        [r["bin"], r["question_type"], int(r.get("n_questions", 0) or 0), *[_fmt_pct(r.get(f"{g}_acc")) for g in rungs]]
        for _, r in df.iterrows()
    ]
    return _md_table(headers, rows)


def _paper_table5(df: pd.DataFrame) -> str:
    headers = ["Bin", "Evidence", "Share %", "Modality frontier", "Bin frontier", "Predicted bin frontier", "Match"]
    rows = [
        [
            r["bin"], r["evidence_modality"], _fmt_pct(r.get("share")),
            r.get("modality_frontier", ""), r.get("bin_frontier", ""), r.get("predicted_bin_frontier", ""),
            "yes" if r.get("predicted_matches_bin") else "no",
        ]
        for _, r in df.iterrows()
    ]
    return _md_table(headers, rows)


def _paper_table6(df: pd.DataFrame) -> str:
    headers = ["Bin", "Pipeline", "Retrieval", "Accuracy [CI]", "Δ vs matched (pts)", "Retrieval F1"]
    rows = []
    for _, r in df.iterrows():
        acc = _fmt_pct(r.get("accuracy"))
        acc_ci = f"{acc} [{_fmt_pct(r.get('ci_low'))}, {_fmt_pct(r.get('ci_high'))}]" if acc else ""
        delta = r.get("delta_accuracy_vs_matched")
        delta_str = f"{float(delta) * 100:+.1f}" if delta is not None and str(delta) != "nan" else ""
        rows.append([r["bin"], r["pipeline"], r.get("retrieval_modality", ""), acc_ci, delta_str, _fmt_pct(r.get("retrieval_f1"))])
    return _md_table(headers, rows)


def _paper_table7(df: pd.DataFrame) -> str:
    headers = ["Policy", "Chosen rungs", "n", "Accuracy [CI]", "Total latency (s)"]
    rows = []
    for _, r in df.iterrows():
        acc = _fmt_pct(r.get("accuracy"))
        acc_ci = f"{acc} [{_fmt_pct(r.get('ci_low'))}, {_fmt_pct(r.get('ci_high'))}]" if acc else ""
        rows.append([r["policy"], r.get("chosen_rungs", ""), int(r.get("n_rows", 0) or 0), acc_ci, f"{float(r.get('total_latency_bs1_s', 0) or 0):.2f}"])
    return _md_table(headers, rows)


_PAPER_RENDERERS: Mapping[str, "Callable[[pd.DataFrame], str]"] = {
    "table1": lambda df: _paper_ladder(df, ["bin"], ["Bin"]),
    "table2": _paper_table2,
    "table3": lambda df: _paper_ladder(df, ["model_spec", "model_size", "bin"], ["Model", "Size", "Bin"], extra=["matches_primary_frontier"]),
    "table4": lambda df: _paper_ladder(df, ["dataset", "bin"], ["Dataset", "Bin"]),
    "table5": _paper_table5,
    "table6": _paper_table6,
    "table7": _paper_table7,
}


def render_paper_tables_markdown(tables: Mapping[str, pd.DataFrame], *, source: str | None = None) -> str:
    """Render the built tables as compact, paper-style markdown (all_tables_summarised.md).

    Only the interpretable columns: document-level accuracy as a percentage with a
    95% bootstrap CI in brackets, the frontier rung in bold. Tables that weren't
    built (unfinished dependency, or table 8's unimplemented scale task) get a
    one-line note instead of a wall of empty columns.
    """

    from datetime import datetime, timezone

    lines = ["# MP-VRDU results (paper tables)", ""]
    meta = [f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}."]
    if source:
        meta.append(f"Source: `{source}`.")
    lines.append(" ".join(meta))
    lines.append("")
    lines.append(
        "Cells are document-level accuracy (%) with a 95% bootstrap CI in [brackets]. "
        "Rungs: T = text, TL = text+layout, TLV = text+layout+vision, V = vision. "
        "The frontier (cheapest sufficient rung) is in **bold**."
    )
    lines.append("")
    for key in TABLE_FILENAMES:
        number = key.removeprefix("table")
        lines.append(f"## Table {number}. {TABLE_TITLES.get(key, key)}")
        lines.append("")
        df = tables.get(key)
        if df is None or df.empty:
            note = (
                "_Not built: scale task (G4) is not implemented._"
                if key == "table8"
                else "_Not built yet: its source experiments' generate/judge haven't all finished._"
            )
            lines.append(note)
        else:
            lines.append(_PAPER_RENDERERS[key](df))
        lines.append("")
    return "\n".join(lines)


def write_all_tables(
    rows: Sequence[ResultRow],
    output_dir: Path,
    *,
    dataset: str = "mmlongbench",
    bins: Sequence[str] = DEFAULT_BINS,
    margin_points: float = 3.0,
    n_bootstrap: int = 1000,
    seed: int = 0,
    markdown_path: Path | None = None,
    markdown_source: str | None = None,
) -> dict[str, Path]:
    """Write all eight table CSV files and return their paths.

    When `markdown_path` is set, also write a single markdown file with all eight
    tables filled in (blank skeletons for tables that have no cached rows).
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = build_all_tables(
        rows,
        dataset=dataset,
        bins=bins,
        margin_points=margin_points,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    paths: dict[str, Path] = {}
    for key, table in tables.items():
        path = output_dir / TABLE_FILENAMES[key]
        table.to_csv(path, index=False)
        paths[key] = path
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            render_tables_markdown(tables, source=markdown_source, n_rows=len(rows)) + "\n"
        )
        paths["markdown"] = markdown_path
    return paths
