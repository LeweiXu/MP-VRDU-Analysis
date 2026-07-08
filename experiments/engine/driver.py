"""Cell-level run primitives: read cell rows, run every cell to exactly one row
regardless of outcome, select the failed rows, and merge a failed-only re-run in
place."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any


def read_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield each jsonl row of a predictions/results file as a dict."""

    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _field(item: Any, name: str, default: Any = None) -> Any:
    """Read a field from a mapping row/cell or a dataclass-like object."""

    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def classify_failure(exc: BaseException) -> tuple[str, str]:
    """Map an exception to a `(status, skipped_reason)` pair.

    A CUDA out-of-memory reads as `oom` (the expected, recoverable failure a
    failed-only re-run on a bigger GPU completes); anything else is `error`.
    """

    reason = f"{type(exc).__name__}: {exc}"
    status = "oom" if "out of memory" in str(exc).lower() else "error"
    return status, reason


def _failed_row(cell: Any, exc: BaseException) -> dict[str, Any]:
    """Default failure row: carry the cell's identity, stamp status + reason."""

    status, reason = classify_failure(exc)
    row = dict(cell) if isinstance(cell, Mapping) else {"prediction_key": _field(cell, "prediction_key")}
    row["status"] = status
    row["skipped_reason"] = reason
    row["oom_occurred"] = status == "oom"
    return row


def run_cells(
    cells: Sequence[Any],
    run_one: Callable[[Any], Any],
    *,
    on_failure: Callable[[Any, BaseException], Any] | None = None,
) -> list[Any]:
    """Run `run_one` over every cell and return one row per cell.

    A cell that succeeds contributes `run_one(cell)`; a cell that raises
    contributes a failure row (via `on_failure`, default `_failed_row`) carrying
    its identity plus `status` in {oom, error} and a `skipped_reason`. The output
    always has exactly `len(cells)` rows, so a failure is data, never a hole.
    """

    build_failed = on_failure or _failed_row
    rows: list[Any] = []
    for cell in cells:
        try:
            rows.append(run_one(cell))
        except Exception as exc:  # noqa: BLE001 - a cell failure is recorded, not raised
            rows.append(build_failed(cell, exc))
    return rows


def select_failed(rows: Sequence[Any]) -> list[Any]:
    """Return the rows whose status is not `ok` (the re-run work queue)."""

    return [row for row in rows if _field(row, "status") != "ok"]


def merge_failed_only(existing: Sequence[Any], reruns: Sequence[Any]) -> list[Any]:
    """Upgrade failed rows in place from a failed-only re-run.

    A row keyed by `prediction_key` that was not `ok` and appears in `reruns` is
    replaced by its re-run; `ok` rows are left untouched. The result has the same
    rows as `existing` (no duplicates), converging the file toward complete.
    """

    reruns_by_key = {_field(r, "prediction_key"): r for r in reruns}
    merged: list[Any] = []
    for row in existing:
        key = _field(row, "prediction_key")
        if _field(row, "status") != "ok" and key in reruns_by_key:
            merged.append(reruns_by_key[key])
        else:
            merged.append(row)
    return merged
