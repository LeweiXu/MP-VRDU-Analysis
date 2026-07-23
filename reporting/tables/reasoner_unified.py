"""One unified reasoner table: precision, scale, matched memory budget, family,
and reasoning variant as blocks sharing the rung columns."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from scoring.accuracy import accuracy_summary
from scoring.frontier import RUNG_ORDER

from ._common import Table, acc_cell, group_by, rows_for_condition
from ._load import column_n_footer, weights_mb

BLANK = "-"
# (block label, specs in block, cell kind). A spec can anchor several blocks;
# the 8B bf16 baseline appears wherever it is the comparison point. Rows are
# emitted only for specs whose rows actually exist in the loaded pool.
BLOCKS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("precision", ("qwen3vl-8b-local", "qwen3vl-8b-local-8bit", "qwen3vl-8b-local-4bit"), "acc"),
    ("scale", ("qwen3vl-2b-local", "qwen3vl-4b-local", "qwen3vl-8b-local", "qwen3vl-32b-local"), "acc"),
    ("matched budget (~17 GB)", ("qwen3vl-8b-local", "qwen3vl-32b-local-4bit"), "acc"),
    ("family", ("qwen3vl-8b-local", "internvl3-8b-local", "llama3.2-11b-vision-local"), "acc"),
    ("reasoning variant (M−S)", ("qwen3vl-8b-local", "qwen3vl-8b-thinking-local"), "ms"),
)
NOTE = (
    "Weight footprint (MB, `~` = derived for quantized variants) replaces peak "
    "VRAM: the measured figure is device-0 only. Every accuracy cell carries its "
    "own n because OOM attrition is not random with respect to the question "
    "pool: it tracks document length and page count, which track the multi-page "
    "questions, so a thin cell compares an easier surviving subset. The "
    "reasoning-variant block reports M−S per rung (multi minus single "
    "accuracy, negative = multi worse), NOT pooled accuracy: the Thinking "
    "variant's value is entirely its hop-split behaviour. Blocks share the 8B "
    "bf16 baseline row wherever it is the comparison point; pool composition "
    "differs by run (scan filters), so compare within a block."
)


def _ms_cell(rows: Sequence[Any]) -> str:
    """Multi minus single accuracy in points, with both sides' n."""

    by_hop = group_by(rows, lambda r: getattr(r, "hop", ""))
    single, multi = by_hop.get("single", []), by_hop.get("multi", [])
    if not single or not multi:
        return BLANK
    delta = accuracy_summary(multi).accuracy - accuracy_summary(single).accuracy
    return f"{delta * 100:+.1f} (nS={len(single)}, nM={len(multi)})"


def build(rows: Sequence[Any]) -> Table:
    """Reasoner blocks x rung, one row per (block, model spec) with data."""

    oracle = rows_for_condition(rows, "oracle")
    if not oracle:
        raise ValueError("reasoner_unified: no oracle rows loaded")
    by_spec = group_by(oracle, lambda r: getattr(r, "model_spec", ""))
    rungs = [r for r in RUNG_ORDER if any(getattr(row, "representation", "") == r for row in oracle)]

    columns = ["block", "model_spec", "weights_mb", *rungs, "n"]
    table_rows: list[list[str]] = []
    for label, specs, kind in BLOCKS:
        for spec in specs:
            spec_rows = by_spec.get(spec, [])
            if not spec_rows:
                continue
            by_rung = group_by(spec_rows, lambda r: getattr(r, "representation", ""))
            if kind == "ms":
                cells = [_ms_cell(by_rung.get(rung, [])) for rung in rungs]
            else:
                cells = [f"{acc_cell(by_rung.get(rung, []))} (n={len(by_rung.get(rung, []))})"
                         for rung in rungs]
            table_rows.append([label, spec, weights_mb(spec), *cells, str(len(spec_rows))])

    return Table(
        key="reasoner_unified",
        title="Reasoner: precision / scale / matched budget / family / reasoning variant",
        columns=columns,
        rows=table_rows,
        note=NOTE,
        footer=column_n_footer(columns, {rung: sum(1 for r in oracle if getattr(r, "representation", "") == rung)
                                         for rung in rungs}),
    )
