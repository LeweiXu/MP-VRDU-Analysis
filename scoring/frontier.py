"""Sufficiency-frontier rule over the representation ladder."""

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
    """Return the cheapest rung whose CI upper bound reaches within the margin.

    Rungs are ordered cheapest to most expensive; the chosen rung is the first
    whose upper CI is within `margin_points` of the strongest rung's accuracy.
    """

    available = [rung for rung in rung_order if rung in cells]
    if not available:
        return ""
    strongest = max(cells[rung].accuracy for rung in available)
    threshold = strongest - _margin_fraction(margin_points)
    for rung in available:
        if cells[rung].ci_high >= threshold:
            return rung
    return available[-1]
