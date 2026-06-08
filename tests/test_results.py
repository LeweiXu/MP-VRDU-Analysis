"""Stage 0 — results writer round-trip (the explicit Stage-0 TEST)."""

from mpvrdu.config import RunConfig
from mpvrdu.results import ResultsWriter, read_rows, read_results, results_path


def test_write_read_roundtrip(tmp_path):
    rows = [
        {"qid": "1", "pred": "a", "correct": True},
        {"qid": "2", "pred": "b", "correct": False},
        {"qid": "3", "pred": "c", "correct": True},
    ]
    out = tmp_path / "out.jsonl"
    with ResultsWriter(out) as w:
        for r in rows:
            w.write(r)
        assert w.count == 3

    back = read_rows(out)
    assert back == rows


def test_meta_header_written(tmp_path):
    cfg = RunConfig(name="t")
    out = tmp_path / "m.jsonl"
    with ResultsWriter(out, config=cfg) as w:
        w.write({"qid": "1"})
    allrows = read_results(out)
    assert allrows[0]["kind"] == "meta"
    assert allrows[0]["config_hash"] == cfg.hash()
    assert read_rows(out) == [{"qid": "1"}]


def test_results_path_encodes_hash():
    cfg = RunConfig(name="my run")
    p = results_path(cfg, results_dir="results", timestamp="20260608-000000")
    assert cfg.hash() in p.name
    assert p.name.startswith("my-run__")
