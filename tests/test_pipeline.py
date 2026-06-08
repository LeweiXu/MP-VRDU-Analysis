"""Stage 2/3 — end-to-end pipeline on the dev slice with mocks.

Proves the SCORING + plumbing are correct independent of model quality:
- mock(gold)  -> ~100% accuracy
- mock(wrong) -> ~0% accuracy
And (Stage 3) the oracle selector feeds EXACTLY the gold pages.
"""

from mpvrdu.config import dict_to_config
from mpvrdu.pipeline import run
from mpvrdu.results import read_rows
from mpvrdu.retrieve.base import Oracle


def _cfg(method, mode, modality="image"):
    return dict_to_config({
        "name": f"{method}-{mode}-{modality}",
        "data": {"name": "synthetic"},
        "representation": {"parser": "pymupdf", "dpi": 72},
        "retrieval": {"method": method, "top_k": 4},
        "generation": {"generator": "mock", "mock_mode": mode, "modality": modality},
        "judge": {"type": "rule"},
    })


def test_mock_gold_scores_perfect(synthetic_ds, chdir_tmp):
    m = run(_cfg("oracle", "gold"), dataset=synthetic_ds, out_path="out.jsonl")
    assert m["accuracy"] == 1.0
    assert m["n"] == 7


def test_mock_wrong_scores_zero(synthetic_ds, chdir_tmp):
    # "wrong" answers everything with a non-abstaining sentinel; unanswerable
    # questions (gold abstains) are therefore also wrong -> 0% overall.
    m = run(_cfg("oracle", "wrong"), dataset=synthetic_ds, out_path="out.jsonl")
    assert m["accuracy"] == 0.0


def test_jsonl_is_wellformed(synthetic_ds, chdir_tmp):
    run(_cfg("oracle", "gold"), dataset=synthetic_ds, out_path="out.jsonl")
    rows = read_rows("out.jsonl")
    assert len(rows) == 7
    required = {"qid", "doc_id", "pred", "correct", "recall_at_k",
                "selected_pages_0based", "question_type"}
    for r in rows:
        assert required <= set(r)


def test_text_modality_runs(synthetic_ds, chdir_tmp):
    # exercises the parser path (text packed from selected pages)
    m = run(_cfg("oracle", "gold", modality="text"), dataset=synthetic_ds,
            out_path="out.jsonl")
    assert m["accuracy"] == 1.0


def test_no_retrieval_runs(synthetic_ds, chdir_tmp):
    cfg = dict_to_config({
        "name": "noretr",
        "data": {"name": "synthetic"},
        "retrieval": {"method": "none", "no_retrieval_pages": 2, "top_k": 2},
        "generation": {"generator": "mock", "mock_mode": "gold", "modality": "image"},
    })
    m = run(cfg, dataset=synthetic_ds, out_path="out.jsonl")
    assert m["n"] == 7
    # first-2-pages selection -> recall < 1 for some questions (e.g. evidence on p3)
    assert m["mean_recall_at_k"] < 1.0


def test_oracle_feeds_exactly_gold_pages(synthetic_ds):
    oracle = Oracle()
    for q in synthetic_ds.questions:
        doc = synthetic_ds.get_document(q.doc_id)
        sel = oracle.select(q, doc)
        assert sel.page_indices == q.evidence_pages_zero_based
        if q.is_unanswerable:
            assert sel.page_indices == []   # nothing fed; generator must abstain
