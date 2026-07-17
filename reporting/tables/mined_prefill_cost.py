"""Mined: prefill cost per rung per doc_type (the clean, decode-free cost axis).

Prefill latency and input-token counts are uncontaminated by the verbose-answer
change, so this is the honest "what does each representation cost to ingest" table,
a cleaner cost story than the decode-inflated end-to-end latency.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from scoring.frontier import RUNG_ORDER

from ._common import Table, base_condition, doc_type_of, group_by, ordered_doc_types, prefill_ms, restrict_to_primary_spec


def _mean_input_tokens(rows: Sequence[Any]) -> float:
    rows = list(rows)
    if not rows:
        return 0.0
    total = sum(int(getattr(r, "total_text_tokens", 0)) + int(getattr(r, "total_visual_tokens", 0)) for r in rows)
    return total / len(rows)


def build(rows: Sequence[Any]) -> Table:
    """doc_type x rung -> mean prefill latency (ms) and mean input tokens."""

    oracle = restrict_to_primary_spec([r for r in rows if base_condition(getattr(r, "condition", "")) == "oracle"] or list(rows))
    present_rungs = [r for r in RUNG_ORDER if any(getattr(x, "representation", "") == r for x in oracle)]

    columns = ["doc_type", "rung", "prefill_ms", "input_tokens", "n"]
    by_doc_type = group_by(oracle, doc_type_of)
    table_rows: list[list[str]] = []
    for dt in ordered_doc_types(oracle):
        for rung in present_rungs:
            group = [r for r in by_doc_type[dt] if getattr(r, "representation", "") == rung]
            if not group:
                continue
            table_rows.append([dt, rung, prefill_ms(group), f"{_mean_input_tokens(group):.0f}", str(len(group))])
    return Table(
        key="mined_prefill_cost",
        title="Mined: prefill cost per rung per doc_type (clean cost axis)",
        columns=columns,
        rows=table_rows,
        note="prefill latency + input tokens are unaffected by the verbose-answer inflation.",
    )


def summary(rows: Sequence[Any]) -> Table:
    """Overall prefill cost per rung, pooled across all doc_types."""

    oracle = restrict_to_primary_spec(
        [r for r in rows if base_condition(getattr(r, "condition", "")) == "oracle"] or list(rows)
    )
    present_rungs = [r for r in RUNG_ORDER if any(getattr(x, "representation", "") == r for x in oracle)]
    columns = ["rung", "prefill_ms", "input_tokens", "n"]
    table_rows: list[list[str]] = []
    for rung in present_rungs:
        group = [r for r in oracle if getattr(r, "representation", "") == rung]
        if not group:
            continue
        table_rows.append([rung, prefill_ms(group), f"{_mean_input_tokens(group):.0f}", str(len(group))])
    return Table(key="prefill_cost_summary", title="Prefill cost (overall): per rung across all doc_types",
                 columns=columns, rows=table_rows,
                 note="prefill latency + input tokens are unaffected by the verbose-answer inflation.")
