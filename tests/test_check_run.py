"""check_run summarizes cell status and flags broken tasks."""
from __future__ import annotations

import json

from conftest import require


def _write(tmp_path, rows):
    p = tmp_path / "results.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_summarize_counts_status_and_reasons(tmp_path):
    summarize = require("ops.scripts.check_run", "summarize_status")
    path = _write(tmp_path, [
        {"status": "ok"}, {"status": "ok"},
        {"status": "oom", "skipped_reason": "CUDA out of memory\nfoo"},
        {"status": "error", "skipped_reason": "ParserCacheMiss: no md"},
        {},  # missing status defaults to ok
    ])
    counts, reasons = summarize(path)
    assert counts["ok"] == 3 and counts["oom"] == 1 and counts["error"] == 1
    assert reasons["CUDA out of memory"] == 1
    assert reasons["ParserCacheMiss: no md"] == 1


def test_verdict_levels(tmp_path):
    from collections import Counter
    verdict = require("ops.scripts.check_run", "verdict")
    assert verdict(Counter(), None, 0.02)[0] == "FAIL"                      # nothing ran
    assert verdict(Counter(ok=100), 100, 0.02)[0] == "OK"                   # all good
    assert verdict(Counter(ok=90, error=10), 100, 0.02)[0] == "FAIL"        # 10% failed
    assert verdict(Counter(ok=99, error=1), 100, 0.02)[0] == "WARN"         # 1% failed, under gate
    assert verdict(Counter(ok=50), 100, 0.02)[0] == "WARN"                  # 50 missing
