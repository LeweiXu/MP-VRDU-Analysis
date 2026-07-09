"""Shared helpers for the table builders: the Table container, row loading, bin
and rung ordering, and per-group accuracy/cost formatting."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from config import DEFAULT_BINS
from scoring.accuracy import accuracy_summary
from scoring.cost import cost_summary
from scoring.frontier import RUNG_ORDER, FrontierCell, sufficiency_frontier

# Bins in thesis order, plus the bucket unlabeled rows fall into (bins are blank
# until the manual annotation pass fills annotations/doc_labels.csv).
BIN_ORDER: tuple[str, ...] = DEFAULT_BINS
UNLABELED = "(unlabeled)"

# Identity of one cell (a prediction without its judge); used to collapse re-runs
# and multi-judge history to a single row per cell before aggregating.
IDENTITY_FIELDS = ("question_id", "doc_id", "condition", "representation", "model_spec")


@dataclass
class Table:
    """A built table: a key, a human title, column headers, and string rows."""

    key: str
    title: str
    columns: list[str]
    rows: list[list[str]]
    note: str = ""


def as_row(data: Any) -> Any:
    """Expose a jsonl dict row through attribute access (what scoring expects)."""

    return SimpleNamespace(**data) if isinstance(data, dict) else data


def read_jsonl(path: str | Path) -> list[Any]:
    """Read a jsonl file into attribute-access row objects (empty if absent)."""

    p = Path(path)
    if not p.exists():
        return []
    return [as_row(json.loads(line)) for line in p.read_text().splitlines() if line.strip()]


def load_ok_rows(path: str | Path) -> list[Any]:
    """Read result rows, collapse each cell to one row, keep only `status == ok`.

    A failed-then-completed cell can appear twice; the `ok` row wins. Scoring
    ignores non-ok rows, so this is where they drop out of the tables.
    """

    best: dict[tuple, Any] = {}
    for row in read_jsonl(path):
        key = tuple(getattr(row, f, "") for f in IDENTITY_FIELDS)
        current = best.get(key)
        if current is None or (getattr(current, "status", "") != "ok" and getattr(row, "status", "") == "ok"):
            best[key] = row
    return [row for row in best.values() if getattr(row, "status", "") == "ok"]


def bin_of(row: Any) -> str:
    """The row's modality bin, bucketed to `(unlabeled)` when blank."""

    return getattr(row, "bin_label", "") or UNLABELED


def ordered_bins(rows: Iterable[Any]) -> list[str]:
    """Present bins in thesis order, unlabeled last."""

    present = {bin_of(row) for row in rows}
    ordered = [b for b in BIN_ORDER if b in present]
    return ordered + sorted(present - set(ordered))


def group_by(rows: Iterable[Any], keyfn) -> dict[Any, list[Any]]:
    """Bucket rows by an arbitrary key function."""

    out: dict[Any, list[Any]] = {}
    for row in rows:
        out.setdefault(keyfn(row), []).append(row)
    return out


def acc_cell(rows: Sequence[Any]) -> str:
    """Format accuracy as `pct [ci_low-ci_high]`, or `-` for no rows."""

    rows = list(rows)
    if not rows:
        return "-"
    s = accuracy_summary(rows)
    return f"{s.accuracy * 100:.1f} [{s.ci_low * 100:.1f}-{s.ci_high * 100:.1f}]"


def frontier_rung(rows: Sequence[Any], *, margin_points: float = 3.0) -> str:
    """The cheapest sufficient rung across a group's rows (empty if none)."""

    cells: dict[str, FrontierCell] = {}
    for rung, group in group_by(rows, lambda r: getattr(r, "representation", "")).items():
        if rung in RUNG_ORDER and group:
            s = accuracy_summary(group)
            cells[rung] = FrontierCell(accuracy=s.accuracy, ci_high=s.ci_high)
    return sufficiency_frontier(cells, margin_points=margin_points) if cells else ""


def latency_ms(rows: Sequence[Any]) -> str:
    """Mean end-to-end latency in milliseconds for a group."""

    rows = list(rows)
    return f"{cost_summary(rows).latency_bs1_s * 1000:.0f}" if rows else "-"


def peak_vram_mb(rows: Sequence[Any]) -> str:
    """Peak VRAM in MB across a group (the binding memory figure)."""

    rows = list(rows)
    return f"{cost_summary(rows).peak_vram_bytes / 1e6:.0f}" if rows else "-"


def write_csv(table: Table, path: str | Path) -> None:
    """Write one table to CSV (header + string rows)."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(table.columns)
        writer.writerows(table.rows)
