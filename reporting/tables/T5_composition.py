"""Table 5: evidence-composition mediation (why the recipe looks that way).

Purpose:
    Decomposes each bin into normalized evidence-source shares (text / table /
    chart / figure / layout), measures a per-modality frontier, and predicts the
    bin's frontier from the strongest modality with >=10% share.
    `predict_frontier_from_composition` is the mediation step, reused by the tests.

Arguments:
    None. Import-only; `build_table5_composition_mediation(rows, ...)` takes rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from config import DEFAULT_BINS
from metrics.frontier import RUNG_ORDER, sufficiency_frontier
from pipeline.orchestrator import ResultRow

from ._common import (
    _normalise_source,
    _rung_metrics,
    _safe_bin,
    _unique_doc_count,
    _unique_question_count,
)


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
