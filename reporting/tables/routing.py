"""Routing-policy comparison, assembled at build time from G1's ladder rows plus
G3's classifier price (routing is not itself a generation task)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from scoring.accuracy import accuracy_summary
from scoring.cost import cost_summary

from ._common import Table, bin_of, frontier_rung, group_by


def _acc_pct(rows: Sequence[Any]) -> float:
    return accuracy_summary(list(rows)).accuracy * 100 if rows else 0.0


def _latency_ms(rows: Sequence[Any]) -> float:
    return cost_summary(list(rows)).latency_bs1_s * 1000 if rows else 0.0


def _oracle_rows(rows: Sequence[Any], *, margin_points: float) -> list[Any]:
    """Rows the oracle policy would keep: each bin's frontier rung."""

    kept: list[Any] = []
    for bin_rows in group_by(rows, bin_of).values():
        chosen = frontier_rung(bin_rows, margin_points=margin_points)
        kept += [r for r in bin_rows if getattr(r, "representation", "") == chosen]
    return kept


def build(rows: Sequence[Any], classifier_rows: Sequence[Any] = (), *, margin_points: float = 3.0) -> Table:
    """Four routing policies: accuracy and mean latency (predicted adds the
    classifier's own latency)."""

    oracle = [r for r in rows if getattr(r, "condition", "") == "oracle"] or list(rows)
    by_rung = group_by(oracle, lambda r: getattr(r, "representation", ""))
    routed = _oracle_rows(oracle, margin_points=margin_points)
    clf_ms = (
        sum(float(getattr(c, "latency_s", 0.0)) for c in classifier_rows) / len(classifier_rows) * 1000
        if classifier_rows
        else 0.0
    )

    policies = [
        ("uniform_cheapest_T", by_rung.get("T", []), 0.0, ""),
        ("uniform_strongest_TLV", by_rung.get("TLV", []), 0.0, ""),
        ("oracle_routing", routed, 0.0, "per-bin frontier rung"),
        ("predicted_routing", routed, clf_ms, "oracle rung choice + classifier latency"),
    ]
    columns = ["policy", "accuracy", "latency_ms", "note"]
    table_rows = [
        [name, f"{_acc_pct(pol_rows):.1f}", f"{_latency_ms(pol_rows) + extra_ms:.0f}", note]
        for name, pol_rows, extra_ms, note in policies
    ]
    return Table(
        key="routing",
        title="Routing policies: accuracy vs latency",
        columns=columns,
        rows=table_rows,
        note="assembled from G1 ladder rows + G3 classifier price",
    )
