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
from scoring.frontier import RUNG_ORDER

from ._common import Table, doc_type_of, group_by, ordered_doc_types, rows_for_condition
from ._load import column_n_footer

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
    candidate = rows_for_condition(rows, "oracle")
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
        footer=column_n_footer(columns, {}),
    )


_SUMMARY_NOTE = (
    "One row per quantization level, so accuracy is directly comparable across the "
    "rungs within a level. Each rung cell is that cell's accuracy with, in "
    "parentheses, its delta against the 16-bit baseline AT THE SAME RUNG, which is "
    "what isolates the quantization effect from the rung mix. "
    "The trailing columns are that level's aggregates over all its rows, not sums or "
    "means of the rung columns. `overall acc` is its pooled correctness rate and "
    "`acc_delta_vs_16bit` compares pooled to pooled; OOM attrition differs by rung "
    "and by level (16-bit TLV survives 717 cells against 4-bit's 762), so the pooled "
    "figures mix slightly different rung compositions and the per-rung cells are the "
    "cleaner comparison. "
    "`vram_mb` is the MAXIMUM peak over the level's rows, not an average, because it "
    "is a headroom figure and the binding cell is what matters. "
    "⚠ It is also SINGLE-DEVICE: peak VRAM is recorded via "
    "`torch.cuda.max_memory_allocated()` with no device argument, so on a "
    "model-parallel load it reports device 0 only and understates the true "
    "footprint. See docs/DECISIONS.md."
)


def _metric_cells(group: Sequence[Any], base_acc: float | None, base_vram: float | None) -> list[str]:
    """Accuracy, peak VRAM, and both deltas for one group of rows."""

    acc = accuracy_summary(list(group)).accuracy * 100
    vram = cost_summary(list(group)).peak_vram_bytes / 1e6
    return [
        f"{acc:.1f}",
        f"{vram:.0f}",
        f"{acc - base_acc:+.1f}" if base_acc is not None else "-",
        f"{vram - base_vram:+.0f}" if base_vram is not None else "-",
        str(len(group)),
    ]


def _rung_of(row: Any) -> str:
    return getattr(row, "representation", "")


def _rung_cell(group: Sequence[Any], base_acc: float | None) -> str:
    """Accuracy for one (quant, rung) cell, with its within-rung delta vs 16-bit."""

    if not group:
        return "-"
    acc = accuracy_summary(list(group)).accuracy * 100
    if base_acc is None:
        return f"{acc:.1f}"
    return f"{acc:.1f} ({acc - base_acc:+.1f})"


def _baseline_metrics(group: Sequence[Any]) -> tuple[float | None, float | None]:
    """The 16-bit accuracy and VRAM a delta is measured against (None when absent)."""

    if not group:
        return None, None
    return accuracy_summary(list(group)).accuracy * 100, cost_summary(list(group)).peak_vram_bytes / 1e6


def summary(rows: Sequence[Any]) -> Table:
    """Accuracy + VRAM per quant level and rung, pooled across all doc_types."""

    candidate = rows_for_condition(rows, "oracle")
    oracle = [r for r in candidate if _base_spec(getattr(r, "model_spec", "")) == DEFAULT_REASONER_SPEC]
    by_quant = group_by(oracle, lambda r: quant_of(getattr(r, "model_spec", "")))
    baseline = by_quant.get("16bit", [])
    by_rung_baseline = group_by(baseline, _rung_of)
    pooled_base_acc, pooled_base_vram = _baseline_metrics(baseline)
    present = [r for r in RUNG_ORDER if any(_rung_of(x) == r for x in oracle)]

    columns = ["quant", *present, "overall acc", "vram_mb (max)", "acc_delta_vs_16bit", "vram_delta_mb", "n"]
    table_rows: list[list[str]] = []
    for quant in sorted(by_quant, key=lambda q: _QUANT_ORDER.get(q, 99)):
        by_rung = group_by(by_quant[quant], _rung_of)
        cells = []
        for rung in present:
            group = by_rung.get(rung, [])
            base_acc, _ = _baseline_metrics(by_rung_baseline.get(rung, []))
            cells.append(_rung_cell(group, base_acc))
        # Aggregates are computed over the quant level's own rows, never summed across
        # the rung columns, which overlap on nothing but do not share a denominator.
        overall = _metric_cells(by_quant[quant], pooled_base_acc, pooled_base_vram)
        table_rows.append([quant, *cells, *overall])

    n_by_col = {rung: sum(1 for r in oracle if _rung_of(r) == rung) for rung in present}
    return Table(key="quantization_summary",
                 title="Quantization sensitivity (overall): accuracy per rung + VRAM per quant",
                 columns=columns, rows=table_rows, note=_SUMMARY_NOTE,
                 footer=column_n_footer(columns, n_by_col))
