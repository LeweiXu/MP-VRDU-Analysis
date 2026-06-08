"""Stage 4/6 — full pipeline with a real retriever + modality flips (mock gen)."""

from mpvrdu.config import dict_to_config
from mpvrdu.pipeline import run
from mpvrdu.results import read_rows


def _cfg(method, modality):
    return dict_to_config({
        "name": f"{method}-{modality}",
        "data": {"name": "synthetic"},
        "representation": {"parser": "pymupdf", "chunking": "page", "dpi": 72},
        "retrieval": {"method": method, "top_k": 2},
        "generation": {"generator": "mock", "mock_mode": "gold", "modality": modality},
    })


def test_bm25_pipeline_image(synthetic_ds, chdir_tmp):
    m = run(_cfg("bm25", "image"), dataset=synthetic_ds, out_path="out.jsonl")
    assert m["n"] == 7
    rows = read_rows("out.jsonl")
    # bm25 should retrieve the evidence page for the keyword-heavy single-page Qs
    s2 = next(r for r in rows if r["qid"] == "s2")
    assert s2["recall_at_k"] == 1.0


def test_modalities_run(synthetic_ds, chdir_tmp):
    # Sub-study C: image / text / both all run on the same fixed pipeline
    for modality in ("image", "text", "both"):
        m = run(_cfg("bm25", modality), dataset=synthetic_ds, out_path=f"{modality}.jsonl")
        assert m["n"] == 7
