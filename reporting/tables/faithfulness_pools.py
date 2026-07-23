"""E5 faithfulness: per prompt mode and rung, accuracy and false abstention on
the answerable pool beside abstention on the unanswerable pool."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from scoring.frontier import RUNG_ORDER

from ._common import Table, acc_cell, group_by, restrict_to_primary_spec
from ._load import column_n_footer, load_ok
from .hallucination import _MODE_ORDER, prompt_mode_of

G3F_RUN_TAG = "g3-faithfulness-full"
G3F_TASK = "G3_hallucination"
BLANK = "-"
NOTE = (
    "The two panels share the prompt mode, rung, reasoner, and abstention "
    "detection rule (delimiter extraction happens at judge time for both "
    "pools). Answerable side: abstaining is an ERROR (the false-abstention "
    "cost); pages are oracle, so the `none` row reproduces the G1 headline "
    "ladder. Unanswerable side: abstaining is correct; pages are the bm25 k=3 "
    "similarity set, so the two panels differ in page selection by design (the "
    "claim is about the instruction, not the retriever). `truncated` counts "
    "cells whose generation hit the decode budget with no EOS "
    "(metadata.output_truncated): a nonzero count under a reasoning mode means "
    "the budget, not the prompt, may explain a drop."
)


def _truncated(rows: Sequence[Any]) -> int:
    return sum(1 for r in rows if (getattr(r, "metadata", None) or {}).get("output_truncated"))


def _abstention_rate(rows: Sequence[Any]) -> str:
    if not rows:
        return BLANK
    abstained = sum(1 for r in rows if getattr(r, "abstained", False))
    return f"{abstained / len(rows) * 100:.1f}"


def build(rows: Sequence[Any]) -> Table:
    """Prompt mode x rung, answerable and unanswerable panels side by side."""

    answerable = restrict_to_primary_spec(list(rows))
    if not answerable:
        raise ValueError("faithfulness_pools: no G4 rows (g4-faithfulness-full not generated/judged yet)")
    unanswerable = restrict_to_primary_spec(load_ok((G3F_RUN_TAG,), G3F_TASK))

    a_by = group_by(answerable, lambda r: (prompt_mode_of(r), getattr(r, "representation", "")))
    u_by = group_by(unanswerable, lambda r: (prompt_mode_of(r), getattr(r, "representation", "")))
    modes = sorted({m for m, _ in list(a_by) + list(u_by)}, key=lambda m: (_MODE_ORDER.get(m, 99), m))
    rungs = [r for r in RUNG_ORDER if any(rung == r for _, rung in list(a_by) + list(u_by))]

    columns = ["prompt_mode", "rung", "answerable acc", "false-abstention (%)",
               "unanswerable abstention (%)", "truncated (A/U)", "n (A/U)"]
    table_rows: list[list[str]] = []
    for mode in modes:
        for rung in rungs:
            a_rows = a_by.get((mode, rung), [])
            u_rows = u_by.get((mode, rung), [])
            if not a_rows and not u_rows:
                continue
            table_rows.append([
                mode, rung,
                acc_cell(a_rows) if a_rows else BLANK,
                _abstention_rate(a_rows),
                _abstention_rate(u_rows),
                f"{_truncated(a_rows)}/{_truncated(u_rows)}",
                f"{len(a_rows)}/{len(u_rows)}",
            ])

    return Table(
        key="faithfulness_pools",
        title="Faithfulness: abstention where evidence is absent vs present, per prompt mode and rung",
        columns=columns,
        rows=table_rows,
        note=NOTE,
        footer=column_n_footer(columns, {"answerable acc": len(answerable),
                                         "unanswerable abstention (%)": len(unanswerable)}),
    )
