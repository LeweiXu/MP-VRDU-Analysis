"""Aggregates token, latency, and VRAM cost across cells."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol


class CostRow(Protocol):
    """Minimal row surface required for cost aggregation."""

    total_text_tokens: int
    total_visual_tokens: int
    output_tokens: int
    latency_s: float
    prefill_latency_s: float
    decode_latency_s: float
    peak_vram_bytes: int


@dataclass(frozen=True)
class CostSummary:
    """Mean latency (with the prefill/decode split) plus token and VRAM cost."""

    n_rows: int
    latency_bs1_s: float
    prefill_s: float
    decode_s: float
    total_text_tokens: int
    total_visual_tokens: int
    output_tokens: int
    total_tokens: int
    peak_vram_bytes: int


def cost_summary(rows: Iterable[CostRow]) -> CostSummary:
    """Return latency/token/VRAM cost summary for a group of rows.

    Latency fields are averaged (batch-size-1 wall clock, plus the prefill/decode
    split); peak VRAM is the max across the group (the binding memory figure).
    """

    materialized = list(rows)
    n = len(materialized)
    text = sum(int(row.total_text_tokens) for row in materialized)
    visual = sum(int(row.total_visual_tokens) for row in materialized)
    output = sum(int(row.output_tokens) for row in materialized)

    def _mean(attr: str) -> float:
        return sum(float(getattr(row, attr)) for row in materialized) / n if n else 0.0

    peak_vram = max((int(row.peak_vram_bytes) for row in materialized), default=0)
    return CostSummary(
        n_rows=n,
        latency_bs1_s=_mean("latency_s"),
        prefill_s=_mean("prefill_latency_s"),
        decode_s=_mean("decode_latency_s"),
        total_text_tokens=text,
        total_visual_tokens=visual,
        output_tokens=output,
        total_tokens=text + visual + output,
        peak_vram_bytes=peak_vram,
    )
