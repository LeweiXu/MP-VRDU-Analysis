"""Abstention on unanswerable questions, by prompt condition: the correct
behaviour is abstaining, so the abstention rate is the score."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from ._common import Table, group_by

_PROMPT = re.compile(r"prompt-(?P<mode>\w+)")


def prompt_mode_of(row: Any) -> str:
    """The prompt mode carried in the condition name (falls back to the whole name)."""

    cond = getattr(row, "condition", "")
    m = _PROMPT.search(cond)
    return m.group("mode") if m else cond


# Present the sweep cheapest-guidance first.
_MODE_ORDER = {"none": 0, "generic": 1, "targeted": 2}


def build(rows: Sequence[Any]) -> Table:
    """One row per prompt condition: abstention rate over unanswerable questions."""

    unanswerable = [r for r in rows if getattr(r, "is_unanswerable", False)] or list(rows)
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
    )
