"""Routes tasks to tables, assembles the routing table at build time, and writes CSV and markdown."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

# Fields that together identify one prediction (a cell without its judge). Rows
# sharing these belong to the same cell, so grouping on them collapses a
# multi-judge or re-run history down to one cell.
IDENTITY_FIELDS = ("question_id", "doc_id", "condition", "representation", "model_spec")


def _field(row: Any, name: str, default: Any = "") -> Any:
    """Read a field from a mapping row or an object row."""

    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def group_key(row: Any) -> tuple:
    """Return the prediction-identity key a row groups under."""

    return tuple(_field(row, name) for name in IDENTITY_FIELDS)


def group_rows(rows: Iterable[Any]) -> dict[tuple, list[Any]]:
    """Group result rows by prediction identity (one group per cell)."""

    groups: dict[tuple, list[Any]] = {}
    for row in rows:
        groups.setdefault(group_key(row), []).append(row)
    return groups


def load_result_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a results jsonl file into a list of row dicts."""

    rows: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# Which task(s) feed which content-named table. Routing is explicit so one task
# can feed several tables; the routing table itself is assembled at build time
# from G1's ladder rows plus the G4 classifier price. The per-table builders in
# `reporting.tables` consume the grouped rows.
TASK_TO_TABLES: Mapping[str, tuple[str, ...]] = {
    "G1_oracle_ladder": ("headline", "parser", "resolution", "scale", "composition", "routing"),
    "G2_retrieval": ("matched_cross", "kdepth", "retrieval_accuracy"),
    "G3_hallucination": ("hallucination",),
    "G4_classifier_pricing": ("routing",),
}


def tables_for_task(task_name: str) -> Sequence[str]:
    """Return the content-named tables a task feeds."""

    return TASK_TO_TABLES.get(task_name, ())
