"""Mined: quantization sensitivity, accuracy and VRAM delta per quant level per doc_type.

Cost-frontier framing (accuracy-per-VRAM): how much accuracy and memory each of
4-bit / 8-bit / 16-bit costs, relative to the 16-bit baseline, for the primary
reasoner. Reuses the cached accuracy/VRAM, never recomputing correctness.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from config import DEFAULT_REASONER_SPEC
from scoring.accuracy import accuracy_summary
from scoring.cost import cost_summary

from ._common import Table, doc_type_of, group_by, ordered_doc_types

_QUANT_ORDER = {"4bit": 0, "8bit": 1, "16bit": 2}


def quant_of(spec: str) -> str:
    if spec.endswith("-4bit"):
        return "4bit"
    if spec.endswith("-8bit"):
        return "8bit"
    return "16bit"


def _base_spec(spec: str) -> str:
    for suffix in ("-4bit", "-8bit"):
        if spec.endswith(suffix):
            return spec[: -len(suffix)]
    return spec


def build(rows: Sequence[Any]) -> Table:
    """doc_type x quant -> accuracy and peak VRAM, plus deltas vs the 16-bit baseline."""

    # Only the primary reasoner's own quant levels, so 4/8/16-bit compare the same model.
    # Conditions carry a prompt suffix (e.g. "oracle__none"), so match the oracle prefix.
    candidate = [r for r in rows if getattr(r, "condition", "").split("__", 1)[0] == "oracle"] or list(rows)
    oracle = [r for r in candidate if _base_spec(getattr(r, "model_spec", "")) == DEFAULT_REASONER_SPEC]
    columns = ["doc_type", "quant", "accuracy", "vram_mb", "acc_delta_vs_16bit", "vram_delta_mb", "n"]
    table_rows: list[list[str]] = []
    for dt in ordered_doc_types(oracle):
        dt_rows = [r for r in oracle if doc_type_of(r) == dt]
        by_quant = group_by(dt_rows, lambda r: quant_of(getattr(r, "model_spec", "")))
        baseline = by_quant.get("16bit", [])
        base_acc = accuracy_summary(baseline).accuracy * 100 if baseline else None
        base_vram = cost_summary(baseline).peak_vram_bytes / 1e6 if baseline else None
        for quant in sorted(by_quant, key=lambda q: _QUANT_ORDER.get(q, 99)):
            group = by_quant[quant]
            acc = accuracy_summary(group).accuracy * 100
            vram = cost_summary(group).peak_vram_bytes / 1e6
            acc_delta = f"{acc - base_acc:+.1f}" if base_acc is not None else "-"
            vram_delta = f"{vram - base_vram:+.0f}" if base_vram is not None else "-"
            table_rows.append([dt, quant, f"{acc:.1f}", f"{vram:.0f}", acc_delta, vram_delta, str(len(group))])
    return Table(
        key="mined_quant_sensitivity",
        title="Mined: quantization sensitivity (accuracy + VRAM delta) by doc_type",
        columns=columns,
        rows=table_rows,
        note="delta is vs the 16-bit baseline of the same model; blank when no baseline is in the cache.",
    )


def summary(rows: Sequence[Any]) -> Table:
    """Overall accuracy + VRAM per quant level, pooled across all doc_types."""

    candidate = [r for r in rows if getattr(r, "condition", "").split("__", 1)[0] == "oracle"] or list(rows)
    oracle = [r for r in candidate if _base_spec(getattr(r, "model_spec", "")) == DEFAULT_REASONER_SPEC]
    by_quant = group_by(oracle, lambda r: quant_of(getattr(r, "model_spec", "")))
    baseline = by_quant.get("16bit", [])
    base_acc = accuracy_summary(baseline).accuracy * 100 if baseline else None
    base_vram = cost_summary(baseline).peak_vram_bytes / 1e6 if baseline else None
    columns = ["quant", "accuracy", "vram_mb", "acc_delta_vs_16bit", "vram_delta_mb", "n"]
    table_rows: list[list[str]] = []
    for quant in sorted(by_quant, key=lambda q: _QUANT_ORDER.get(q, 99)):
        group = by_quant[quant]
        acc = accuracy_summary(group).accuracy * 100
        vram = cost_summary(group).peak_vram_bytes / 1e6
        acc_delta = f"{acc - base_acc:+.1f}" if base_acc is not None else "-"
        vram_delta = f"{vram - base_vram:+.0f}" if base_vram is not None else "-"
        table_rows.append([quant, f"{acc:.1f}", f"{vram:.0f}", acc_delta, vram_delta, str(len(group))])
    return Table(key="quantization_summary", title="Quantization sensitivity (overall): accuracy + VRAM per quant",
                 columns=columns, rows=table_rows, note="delta vs the 16-bit baseline of the same model.")
