"""Stage 7 — aggregation tables are a reproducible function of the JSONL."""

import pytest

from mpvrdu.analysis import aggregate_dir, summarize_run, to_markdown_table
from mpvrdu.config import dict_to_config
from mpvrdu.pipeline import run


def _cfg(name, mode):
    return dict_to_config({
        "name": name,
        "data": {"name": "synthetic"},
        "representation": {"parser": "pymupdf", "dpi": 72},
        "retrieval": {"method": "oracle", "top_k": 4},
        "generation": {"generator": "mock", "mock_mode": mode, "modality": "image"},
    })


def _make_results(tmp):
    rdir = tmp / "results"
    run(_cfg("good", "gold"), out_path=rdir / "good.jsonl")
    run(_cfg("bad", "wrong"), out_path=rdir / "bad.jsonl")
    return rdir


def test_summarize_run_breakdowns(synthetic_ds, chdir_tmp):
    run(_cfg("good", "gold"), dataset=synthetic_ds, out_path="good.jsonl")
    s = summarize_run("good.jsonl")
    assert s["n"] == 7
    assert s["accuracy"] == 1.0
    assert s["condition"]["retrieval.method"] == "oracle"
    # all three question types present in the breakdown
    assert set(s["by_question_type"]) == {"single", "cross", "unanswerable"}
    assert s["by_question_type"]["single"]["accuracy"] == 1.0


def test_aggregate_dir_and_table(chdir_tmp):
    rdir = _make_results(chdir_tmp)
    summaries = aggregate_dir(rdir)
    assert len(summaries) == 2
    by_name = {s["name"]: s for s in summaries}
    assert by_name["good"]["accuracy"] == 1.0
    assert by_name["bad"]["accuracy"] == 0.0

    table = to_markdown_table(summaries)
    assert "accuracy" in table and "retrieval.method" in table
    assert table.count("\n") >= 3                 # header + sep + 2 rows


def test_reproducible(chdir_tmp):
    rdir = _make_results(chdir_tmp)
    a = aggregate_dir(rdir)
    b = aggregate_dir(rdir)
    assert a == b                                  # pure function of the JSONL


def test_figures_optional(chdir_tmp):
    pytest.importorskip("matplotlib")
    from mpvrdu.analysis.figures import method_bars, topk_curve

    p1 = topk_curve({"bm25": [(1, 0.5), (4, 0.8)]}, "recall@k", "fig/curve.png")
    p2 = method_bars({"bm25": 0.4, "dense": 0.5}, "accuracy", "fig/bars.png",
                     floor=0.2, ceiling=0.7)
    assert p1.exists() and p2.exists()
