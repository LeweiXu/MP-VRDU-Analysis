"""Select sufficiency frontiers for representation trade-offs.

Purpose:
    Implements the pre-registered Stage-M5 frontier rule: choose the cheapest
    representation rung whose CI upper bound reaches within the configured
    margin of the strongest rung's point estimate.

Pipeline role:
    Table builders apply this rule to each doc-type bin. `T`, `TL`, `TLV`, `V`
    are ordered cheapest to most visually expensive; the function is independent
    of table code so margin sensitivity can reuse it later.

Arguments:
    None. This module is import-only; callers use `sufficiency_frontier()`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


RUNG_ORDER: tuple[str, ...] = ("T", "TL", "TLV", "V")


@dataclass(frozen=True)
class FrontierCell:
    """Accuracy point estimate plus upper CI for one rung."""

    accuracy: float
    ci_high: float


def _margin_fraction(margin_points: float) -> float:
    """Convert accuracy points to a 0-1 fraction."""

    return margin_points / 100.0 if margin_points > 1 else margin_points


def sufficiency_frontier(
    cells: Mapping[str, FrontierCell],
    *,
    margin_points: float = 3.0,
    rung_order: Sequence[str] = RUNG_ORDER,
) -> str:
    """Return the cheapest rung satisfying the sufficiency-frontier rule."""

    available = [rung for rung in rung_order if rung in cells]
    if not available:
        return ""
    strongest = max(cells[rung].accuracy for rung in available)
    threshold = strongest - _margin_fraction(margin_points)
    for rung in available:
        if cells[rung].ci_high >= threshold:
            return rung
    return available[-1]
