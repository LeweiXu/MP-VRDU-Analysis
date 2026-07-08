"""Cell-level robustness: exactly one row per cell regardless of outcome, and
--failed-only re-runs only the failed rows and upgrades them in place."""

from __future__ import annotations

import pytest

from conftest import require


def test_one_row_per_cell_even_on_failure() -> None:
    # A task over N cells emits exactly N rows. A cell that raises still writes a
    # row with status in {oom, error} + skipped_reason, never omitted.
    run_cells = require("experiments.engine.driver", "run_cells")

    cells = [{"prediction_key": f"k{i}"} for i in range(5)]

    def run_one(cell):
        if cell["prediction_key"] in {"k1", "k3"}:
            raise RuntimeError("CUDA out of memory")
        return {"prediction_key": cell["prediction_key"], "status": "ok"}

    rows = run_cells(cells, run_one)
    assert len(rows) == len(cells), "one row per cell must always hold"

    by_key = {r["prediction_key"]: r for r in rows}
    for failed in ("k1", "k3"):
        assert by_key[failed]["status"] in {"oom", "error"}
        assert by_key[failed].get("skipped_reason"), "failed row needs skipped_reason"
    for ok in ("k0", "k2", "k4"):
        assert by_key[ok]["status"] == "ok"


def test_select_failed_picks_non_ok_rows() -> None:
    select_failed = require("experiments.engine.driver", "select_failed")
    rows = [
        {"prediction_key": "a", "status": "ok"},
        {"prediction_key": "b", "status": "oom"},
        {"prediction_key": "c", "status": "error"},
        {"prediction_key": "d", "status": "ok"},
    ]
    failed = select_failed(rows)
    assert {r["prediction_key"] for r in failed} == {"b", "c"}


def test_failed_only_upgrades_in_place() -> None:
    # A supervisor re-run reads a run's rows, re-runs status != ok, and upgrades
    # them in place; ok rows are left untouched (same file converges to complete).
    merge = require("experiments.engine.driver", "merge_failed_only")
    existing = [
        {"prediction_key": "a", "status": "ok", "text": "keep"},
        {"prediction_key": "b", "status": "oom", "text": ""},
    ]
    reruns = [{"prediction_key": "b", "status": "ok", "text": "recovered"}]
    merged = {r["prediction_key"]: r for r in merge(existing, reruns)}
    assert merged["a"]["text"] == "keep", "ok rows must not be re-run"
    assert merged["b"]["status"] == "ok" and merged["b"]["text"] == "recovered"
    assert len(merged) == 2, "no duplicate rows after upgrade"
