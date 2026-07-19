"""Abstention on unanswerable questions, by prompt condition: the correct
behaviour is abstaining, so the abstention rate is the score."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._common import Table, group_by, split_condition, unanswerable_rows
from ._load import column_n_footer


def prompt_mode_of(row: Any) -> str:
    """The prompt mode carried as the condition's `__<mode>` suffix.

    Falls back to the whole condition when it carries no suffix.
    """

    cond = getattr(row, "condition", "")
    _, mode = split_condition(cond)
    return mode or cond


# Present the sweep cheapest-guidance first.
_MODE_ORDER = {"none": 0, "generic": 1, "targeted": 2}


def build(rows: Sequence[Any]) -> Table:
    """One row per prompt condition: abstention rate over unanswerable questions."""

    unanswerable = unanswerable_rows(rows)
    columns = ["prompt_condition", "abstention_rate", "answered", "n"]
    by_mode = group_by(unanswerable, prompt_mode_of)
    table_rows: list[list[str]] = []
    for mode in sorted(by_mode, key=lambda m: (_MODE_ORDER.get(m, 99), m)):
        group = by_mode[mode]
        abstained = sum(1 for r in group if getattr(r, "abstained", False))
        rate = abstained / len(group) if group else 0.0
        table_rows.append([mode, f"{rate * 100:.1f}", str(len(group) - abstained), str(len(group))])
    return Table(
        key="hallucination",
        title="Hallucination: abstention rate on unanswerable questions by prompt",
        columns=columns,
        rows=table_rows,
        footer=column_n_footer(columns, {}),
    )
