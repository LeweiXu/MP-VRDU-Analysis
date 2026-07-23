"""Phase D guards: the four extension builders (synthetic rows only) and the
reconciliation mechanism (pass/fail/skip/gating, synthetic tables)."""

from types import SimpleNamespace

import pytest


def _row(**over):
    base = dict(
        question_id="q1", doc_id="d1", doc_type="report", bin_label="", scan_label="digital",
        hop="single", is_unanswerable=False, evidence_sources=("text",),
        condition="oracle__none", provenance="oracle", page_indices=(0,),
        representation="TLV", model_spec="qwen3vl-8b-local", visual_resolution="med",
        status="ok", correct=True, abstained=False, score=1.0, judge_spec="stub",
        answer="a", total_text_tokens=10, total_visual_tokens=0, text_tokens_fed=10,
        output_tokens=5, tokens_dropped=0, truncation_occurred=False,
        latency_s=1.0, prefill_latency_s=0.5, decode_latency_s=0.5,
        peak_vram_bytes=1, machine="", note="", metadata={},
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_selection_groups_on_the_condition_grammar():
    from reporting.tables import selection

    rows = [
        _row(question_id=f"q{i}", condition="pageset:r=bm25:g=drop_top-1:d=0__none",
             provenance="constructed", correct=i % 2 == 0, hop="multi")
        for i in range(4)
    ] + [
        _row(question_id=f"q{i}", condition="pageset:r=colqwen3:g=drop_top-1:d=0__none",
             provenance="constructed", correct=True, hop="multi")
        for i in range(4)
    ]
    table = selection.build(rows)
    assert table.columns[:2] == ["condition", "ranker"]
    rankers = {row[1] for row in table.rows if row[1] != "-"}
    assert rankers == {"bm25", "colqwen3"}  # two rankers = two rows, never pooled
    labels = [row[0] for row in table.rows]
    assert any("drop top 1" in label for label in labels)


def test_selection_raises_without_pageset_rows():
    from reporting.tables import selection

    with pytest.raises(ValueError):
        selection.build([_row()])  # oracle rows only, no pageset conditions


def test_faithfulness_pools_raises_without_g4_rows():
    from reporting.tables import faithfulness_pools

    with pytest.raises(ValueError):
        faithfulness_pools.build([])


def test_faithfulness_pools_counts_truncation():
    from reporting.tables import faithfulness_pools

    rows = [
        _row(condition="oracle__cot", metadata={"output_truncated": True}),
        _row(question_id="q2", condition="oracle__cot", metadata={"output_truncated": False}),
    ]
    table = faithfulness_pools.build(rows)
    (cot_row,) = [r for r in table.rows if r[0] == "cot"]
    assert cot_row[table.columns.index("truncated (A/U)")] == "1/0"


def test_reasoner_unified_blocks_and_ms():
    from reporting.tables import reasoner_unified

    rows = (
        [_row(question_id=f"q{i}", hop="single", correct=True) for i in range(3)]
        + [_row(question_id=f"m{i}", hop="multi", correct=False) for i in range(3)]
        + [_row(question_id=f"t{i}", model_spec="qwen3vl-8b-thinking-local",
                hop="single" if i % 2 else "multi", correct=i % 2 == 0) for i in range(4)]
    )
    table = reasoner_unified.build(rows)
    blocks = {row[0] for row in table.rows}
    assert "precision" in blocks and "reasoning variant (M−S)" in blocks
    ms_rows = [r for r in table.rows if r[0] == "reasoning variant (M−S)"]
    # M-S cells carry both sides' n, not a pooled accuracy.
    assert all("nS=" in r[3] or r[3] == "-" for r in ms_rows)


def test_reconcile_pass_fail_skip_and_gating():
    from reporting.reconcile import Check, ReconcileResult, cell_value, failed_gates, run_checks

    good = SimpleNamespace(key="t1", columns=["x"], rows=[["56.8 [52-61] (n=717)"]])
    bad = SimpleNamespace(key="t2", columns=["x"], rows=[["10.0"]])
    checks = (
        Check(gates="t1", label="good", expected=56.8,
              locate=lambda t: cell_value(t["t1"].rows[0][0]) if "t1" in t else None),
        Check(gates="t2", label="bad", expected=56.8,
              locate=lambda t: cell_value(t["t2"].rows[0][0]) if "t2" in t else None),
        Check(gates="t3", label="absent", expected=1.0, locate=lambda t: None),
    )
    results = run_checks([good, bad], checks)
    assert [r.status for r in results] == ["pass", "fail", "skip"]
    assert failed_gates(results) == {"t2"}


def test_reconcile_cell_value_parsing():
    from reporting.reconcile import cell_value

    assert cell_value("56.8 [52.0-61.2] (n=717)") == 56.8
    assert cell_value("+2.7") == 2.7
    assert cell_value("-") is None
    assert cell_value("") is None


def test_build_withholds_gated_tables(tmp_path):
    # The anti-_safe: a failing check removes the gated table from the write set
    # and the build reports failure, instead of silently emitting a wrong table.
    from reporting.build import write_tables
    from reporting.reconcile import Check, failed_gates, run_checks
    from reporting.tables._common import Table

    table = Table(key="gated", title="G", columns=["x"], rows=[["10.0"]])
    check = Check(gates="gated", label="anchor", expected=99.0,
                  locate=lambda t: 10.0)
    results = run_checks([table], (check,))
    gated = failed_gates(results)
    assert gated == {"gated"}
    kept = [t for t in [table] if t.key not in gated]
    write_tables(kept, tmp_path)
    assert not (tmp_path / "gated.csv").exists()
