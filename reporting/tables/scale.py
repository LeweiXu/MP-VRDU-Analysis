"""Model-size and quantization cost frontier: accuracy against VRAM and latency
across reasoner specs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from scoring.frontier import RUNG_ORDER

from ._common import (
    SINGLE_DEVICE_VRAM_NOTE,
    Table,
    acc_cell,
    group_by,
    latency_ms,
    peak_vram_mb,
    prefill_ms,
    rows_for_condition,
)
from ._load import column_n_footer


def build(rows: Sequence[Any]) -> Table:
    """One row per reasoner spec: per-rung accuracy plus peak VRAM and latency.

    `latency_ms` is end-to-end and decode-inflated (~20x by the verbose-answer
    change); `prefill_ms` and `peak_vram_mb` are the clean, uncontaminated cost
    signals. This is the one G1 table that keeps every model_spec separate (it is
    the model-size / quantization sweep), so it never restricts to one reasoner.
    """

    oracle = rows_for_condition(rows, "oracle")
    present = [r for r in RUNG_ORDER if any(getattr(x, "representation", "") == r for x in oracle)]
    columns = ["model_spec", *present, "peak_vram_mb", "prefill_ms", "latency_ms", "n"]
    by_spec = group_by(oracle, lambda r: getattr(r, "model_spec", ""))
    table_rows: list[list[str]] = []
    for spec in sorted(by_spec):
        spec_rows = by_spec[spec]
        by_rung = group_by(spec_rows, lambda r: getattr(r, "representation", ""))
        cells = [acc_cell(by_rung.get(rung, [])) for rung in present]
        table_rows.append([spec, *cells, peak_vram_mb(spec_rows), prefill_ms(spec_rows),
                           latency_ms(spec_rows), str(len(spec_rows))])
    by_rung_all = group_by(oracle, lambda r: getattr(r, "representation", ""))
    footer = column_n_footer(columns, {rung: len(by_rung_all.get(rung, [])) for rung in present})
    return Table(
        key="scale",
        title="Scale: accuracy vs VRAM/latency across reasoner specs",
        columns=columns,
        rows=table_rows,
        note=("latency_ms is end-to-end and decode-inflated (~20x by the verbose-answer "
              "change); prefill_ms is the clean cost signal, and reads `-` for a backend "
              "that cannot measure a prefill/decode split. " + SINGLE_DEVICE_VRAM_NOTE),
        footer=footer,
    )
