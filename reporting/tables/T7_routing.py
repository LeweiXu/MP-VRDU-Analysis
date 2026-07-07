"""Table 7: routing policies (what to do without doc-type labels).

Purpose:
    Prices four corpus-level policies (oracle routing, predicted routing,
    uniform-cheapest T, uniform-strongest TLV) on accuracy and total latency.
    Routing accuracy reuses G1's ladder rows and the per-bin recipe from Table 1;
    predicted routing folds in G6's classifier latency (total time / eval rows).

Arguments:
    None. Import-only; `build_table7_routing(rows, classifier_records=...)`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from config import DEFAULT_BINS
from metrics.accuracy import accuracy_summary
from metrics.cost import cost_summary
from pipeline.orchestrator import ResultRow

from ._common import ROUTING_POLICIES, _safe_bin, _selected_row_map
from .T1_headline import build_table1_headline


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
