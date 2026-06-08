"""Experiment-suite expansion: axes -> validated RunConfigs."""

from mpvrdu.experiment import expand_suite, load_suite


def test_expand_cross_product():
    suite = {
        "defaults": {
            "data": {"name": "synthetic"},
            "generation": {"generator": "mock", "mock_mode": "gold"},
        },
        "substudies": {
            "A": {"axes": {
                "retrieval.method": ["bm25", "tfidf"],
                "generation.modality": ["image", "text"],
            }},
        },
    }
    runs = expand_suite(suite)
    assert len(runs) == 4                       # 2 x 2
    subs = {s for s, _ in runs}
    assert subs == {"A"}
    # defaults applied + axes overrides set
    by_name = {c.name: c for _, c in runs}
    assert "A__bm25__image" in by_name
    c = by_name["A__bm25__text"]
    assert c.retrieval.method == "bm25"
    assert c.generation.modality == "text"
    assert c.generation.generator == "mock"     # from defaults


def test_distinct_hashes():
    suite = {
        "defaults": {"data": {"name": "synthetic"}},
        "substudies": {"S": {"axes": {"retrieval.top_k": [1, 2, 4]}}},
    }
    runs = expand_suite(suite)
    hashes = {c.hash() for _, c in runs}
    assert len(hashes) == 3                      # each k -> distinct config/file


def test_kaya_switch_in_defaults():
    # changing defaults.generation flips every run's generator (the "1-line" switch)
    suite = {
        "defaults": {"data": {"name": "synthetic"},
                     "generation": {"generator": "kaya_vlm",
                                    "model_id": "Qwen/Qwen2.5-VL-7B-Instruct"}},
        "substudies": {"A": {"axes": {"retrieval.method": ["bm25", "dense"]}}},
    }
    runs = expand_suite(suite)
    assert all(c.generation.generator == "kaya_vlm" for _, c in runs)
