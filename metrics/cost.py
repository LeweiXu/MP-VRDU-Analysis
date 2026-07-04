"""Aggregate latency and token costs for representation/model comparisons.

Purpose:
    Implements the Stage-M5 cost surface. Latency at batch size 1 is the primary
    paper cost metric; split text/vision token totals remain available for
    secondary accounting and routing-policy tables.

Pipeline role:
    Table builders consume cached `ResultRow` objects and call `cost_summary()`
    for each group. The helper intentionally does no grouping itself so callers
    can slice by bin, condition, model family, or policy.

Arguments:
    None. This module is import-only; callers use `cost_summary(rows)`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol


class CostRow(Protocol):
    """Minimal row surface required for cost aggregation."""

    input_text_tokens: int
    input_visual_tokens: int
    output_tokens: int
    latency_s: float


@dataclass(frozen=True)
class CostSummary:
    """Mean latency plus aggregate split token counts."""

    n_rows: int
    latency_bs1_s: float
    input_text_tokens: int
    input_visual_tokens: int
    output_tokens: int
    total_tokens: int


def cost_summary(rows: Iterable[CostRow]) -> CostSummary:
    """Return latency/token cost summary for a group of rows."""

    materialized = list(rows)
    text = sum(int(row.input_text_tokens) for row in materialized)
    visual = sum(int(row.input_visual_tokens) for row in materialized)
    output = sum(int(row.output_tokens) for row in materialized)
    latency = (
        sum(float(row.latency_s) for row in materialized) / len(materialized)
        if materialized
        else 0.0
    )
    return CostSummary(
        n_rows=len(materialized),
        latency_bs1_s=latency,
        input_text_tokens=text,
        input_visual_tokens=visual,
        output_tokens=output,
        total_tokens=text + visual + output,
    )
