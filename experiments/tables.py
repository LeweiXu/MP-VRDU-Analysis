"""Aggregate cached result rows into the paper's CSV table shapes.

Purpose:
    Implements the Stage-M5 reporting surface. The builders consume judged
    `ResultRow` objects and emit the eight table shapes required by the paper,
    filled with whatever smoke/full rows are currently cached.

Pipeline role:
    `cli.build_tables` calls this module after experiment runs have filled
    `results/cache/`. Keeping aggregation here lets tables be rebuilt without
    rerunning models, judges, parsers, or retrieval.

Arguments:
    None. This module is import-only; callers pass result rows to
    `build_all_tables()` or call `write_all_tables()`.
"""

from __future__ import annotations

import json
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

QUESTION_TYPES = ("none", "single", "multi")
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
    """Build Table 2: bin x question-type analytical slice."""

    out: list[dict[str, object]] = []
    for bin_name in bins:
        for question_type in QUESTION_TYPES:
            group_rows = [row for row in rows if _bin(row) == bin_name and row.hop == question_type]
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
    """Build Table 3: model-family/size replication skeleton."""

    out: list[dict[str, object]] = []
    model_specs = sorted({row.model_spec for row in rows}) or [""]
    for model_spec in model_specs:
        model_rows = [row for row in rows if row.model_spec == model_spec]
        for bin_name in bins:
            group_rows = [row for row in model_rows if _bin(row) == bin_name]
            columns, cells, _ = _rung_metrics(group_rows, n_bootstrap=n_bootstrap, seed=seed)
            out.append(
                {
                    "model_spec": model_spec,
                    "model_size": _size_label(model_spec),
                    "bin": bin_name,
                    "n_questions": _unique_question_count(group_rows),
                    **columns,
                    "frontier": sufficiency_frontier(cells, margin_points=margin_points),
                }
            )
    return pd.DataFrame(out)


def build_table4_dataset_replication(
    rows: Sequence[ResultRow],
    *,
    dataset: str = "mmlongbench",
    bins: Sequence[str] = DEFAULT_BINS,
    margin_points: float = 3.0,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Build Table 4: dataset replication skeleton."""

    table1 = build_table1_headline(
        rows,
        bins=bins,
        margin_points=margin_points,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    table1.insert(0, "dataset", dataset)
    return table1


def build_table5_composition_mediation(
    rows: Sequence[ResultRow],
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Build Table 5: evidence-composition mediation skeleton."""

    sources = sorted({source for row in rows for source in row.evidence_sources}) or ["all"]
    out: list[dict[str, object]] = []
    for source in sources:
        group_rows = [row for row in rows if source == "all" or source in row.evidence_sources]
        columns, _, _ = _rung_metrics(group_rows, n_bootstrap=n_bootstrap, seed=seed)
        out.append(
            {
                "evidence_source": source,
                "n_questions": _unique_question_count(group_rows),
                "n_docs": _unique_doc_count(group_rows),
                **columns,
            }
        )
    return pd.DataFrame(out)


def build_table6_matched_vs_cross(
    rows: Sequence[ResultRow],
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Build Table 6: matched-vs-cross retrieval/reasoning skeleton."""

    definitions = {
        "matched_vision": [
            row
            for row in rows
            if row.condition.startswith("retrieved_vision") and row.representation in {"TLV", "V"}
        ],
        "cross_text_to_vision": [
            row
            for row in rows
            if row.condition.startswith("retrieved_text") and row.representation in {"TLV", "V"}
        ],
    }
    out: list[dict[str, object]] = []
    for pipeline, group_rows in definitions.items():
        acc = accuracy_summary(group_rows, n_bootstrap=n_bootstrap, seed=seed)
        cost = cost_summary(group_rows)
        out.append(
                {
                    "pipeline": pipeline,
                    "condition": ",".join(sorted({row.condition for row in group_rows})) if group_rows else "",
                    "retrieval_modality": "vision" if pipeline == "matched_vision" else "text",
                    "reasoning_modality": "vision",
                    "n_rows": acc.n_rows,
                    "n_docs": acc.n_docs,
                    "accuracy": acc.accuracy,
                    "ci_low": acc.ci_low,
                    "ci_high": acc.ci_high,
                "latency_bs1_s": cost.latency_bs1_s,
            }
        )
    return pd.DataFrame(out)


def build_table7_routing(
    rows: Sequence[ResultRow],
    *,
    bins: Sequence[str] = DEFAULT_BINS,
    margin_points: float = 3.0,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Build Table 7: routing-policy skeleton."""

    out: list[dict[str, object]] = []
    table1 = build_table1_headline(
        rows,
        bins=bins,
        margin_points=margin_points,
        n_bootstrap=n_bootstrap,
        seed=seed,
    ).set_index("bin")
    for bin_name in bins:
        frontier = str(table1.loc[bin_name, "frontier"]) if bin_name in table1.index else ""
        policies = {
            "oracle_routing": frontier,
            "predicted_routing": frontier,
            "uniform_cheapest_T": "T",
            "uniform_strongest_TLV": "TLV",
        }
        for policy in ROUTING_POLICIES:
            rung = policies[policy]
            group_rows = [
                row for row in rows if _bin(row) == bin_name and row.representation == rung
            ] if rung else []
            acc = accuracy_summary(group_rows, n_bootstrap=n_bootstrap, seed=seed)
            cost = cost_summary(group_rows)
            classifier_latency = 0.0
            out.append(
                {
                    "policy": policy,
                    "bin": bin_name,
                    "chosen_rung": rung,
                    "n_rows": acc.n_rows,
                    "accuracy": acc.accuracy,
                    "ci_low": acc.ci_low,
                    "ci_high": acc.ci_high,
                    "latency_bs1_s": cost.latency_bs1_s,
                    "classifier_latency_bs1_s": classifier_latency,
                    "total_latency_bs1_s": cost.latency_bs1_s + classifier_latency,
                    "text_tokens": cost.input_text_tokens,
                    "vision_tokens": cost.input_visual_tokens,
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
            dataset=dataset,
            bins=bins,
            margin_points=margin_points,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
        "table5": build_table5_composition_mediation(rows, n_bootstrap=n_bootstrap, seed=seed),
        "table6": build_table6_matched_vs_cross(rows, n_bootstrap=n_bootstrap, seed=seed),
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


def write_all_tables(
    rows: Sequence[ResultRow],
    output_dir: Path,
    *,
    dataset: str = "mmlongbench",
    bins: Sequence[str] = DEFAULT_BINS,
    margin_points: float = 3.0,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, Path]:
    """Write all eight table CSV files and return their paths."""

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
    return paths
