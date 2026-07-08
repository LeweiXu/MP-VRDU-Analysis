"""Table 6: matched vs cross retrieval (does retrieval need the reasoning modality?).

Purpose:
    On the bins where reasoning materially benefits from vision (best TLV/V beats
    best T/TL by the margin, read off G1's oracle frontier), compare *matched*
    (vision-retrieval -> vision reasoning) against *cross* (text-retrieval -> vision
    reasoning) under G5's real retrieval, one matched/cross pair per k in the sweep.

Arguments:
    None. Import-only; `build_table6_matched_vs_cross(rows, retrieval_records=...)`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import pandas as pd

from config import DEFAULT_BINS
from metrics.accuracy import accuracy_summary
from metrics.cost import cost_summary
from pipeline.orchestrator import ResultRow

from ._common import _safe_bin
from .T1_headline import build_table1_headline


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
