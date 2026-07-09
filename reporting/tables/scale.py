"""Model-size and quantization cost frontier: accuracy against VRAM and latency
across reasoner specs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from scoring.frontier import RUNG_ORDER

from ._common import Table, acc_cell, group_by, latency_ms, peak_vram_mb


def build(rows: Sequence[Any]) -> Table:
    """One row per reasoner spec: per-rung accuracy plus peak VRAM and latency."""

    oracle = [r for r in rows if getattr(r, "condition", "") == "oracle"] or list(rows)
    present = [r for r in RUNG_ORDER if any(getattr(x, "representation", "") == r for x in oracle)]
    columns = ["model_spec", *present, "peak_vram_mb", "latency_ms", "n"]
    by_spec = group_by(oracle, lambda r: getattr(r, "model_spec", ""))
    table_rows: list[list[str]] = []
    for spec in sorted(by_spec):
        spec_rows = by_spec[spec]
        by_rung = group_by(spec_rows, lambda r: getattr(r, "representation", ""))
        cells = [acc_cell(by_rung.get(rung, [])) for rung in present]
        table_rows.append([spec, *cells, peak_vram_mb(spec_rows), latency_ms(spec_rows), str(len(spec_rows))])
    return Table(
        key="scale",
        title="Scale: accuracy vs VRAM/latency across reasoner specs",
        columns=columns,
        rows=table_rows,
    )
