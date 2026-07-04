"""Test the Section-F1 frontier-divergence gate.

Purpose:
    Verifies the first full-run go/no-go predicate: Table 1 passes only when at
    least two configured doc-type bins have different sufficiency frontiers. This
    protects the paper's first human checkpoint from ad hoc spreadsheet logic.

Test role:
    Uses constructed frontier maps and a tiny Table-1 CSV, so the gate can be
    checked without running Qwen, judges, or table builders.

Arguments:
    None. Run with `python -m pytest tests/test_frontier_gate.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.gates import frontier_divergence_gate, load_table1_frontiers


def test_frontier_gate_returns_no_go_when_all_bins_match() -> None:
    result = frontier_divergence_gate(
        {"text_heavy": "TLV", "in_between": "TLV", "visual_heavy": "TLV"}
    )

    assert result.status == "no_go"
    assert not result.passed
    assert result.metric == 1.0


def test_frontier_gate_returns_go_when_two_bins_differ() -> None:
    result = frontier_divergence_gate(
        {"text_heavy": "T", "in_between": "TLV", "visual_heavy": "TLV"}
    )

    assert result.status == "go"
    assert result.passed
    assert result.metric == 2.0
    assert result.details["frontiers"]["text_heavy"] == "T"


def test_frontier_gate_rejects_missing_bin_frontier() -> None:
    with pytest.raises(ValueError, match="missing frontier"):
        frontier_divergence_gate({"text_heavy": "T", "in_between": "TLV"})


def test_load_table1_frontiers(tmp_path: Path) -> None:
    path = tmp_path / "table1.csv"
    path.write_text(
        "bin,frontier,latency_at_frontier_s\n"
        "text_heavy,T,0.1\n"
        "in_between,TL,0.2\n"
        "visual_heavy,TLV,0.3\n"
    )

    assert load_table1_frontiers(path) == {
        "text_heavy": "T",
        "in_between": "TL",
        "visual_heavy": "TLV",
    }
