"""Test Stage-M1 Option-A binning, smoke corpus, and config knobs.

Purpose:
    Verifies all native MMLongBench doc types map to the intended v3 bins, the
    frozen smoke corpus covers every bin, and `ExperimentConfig(smoke=True)`
    keeps paths root-relative while selecting smoke settings.

Test role:
    Protects the stable target that all later MVP smoke stages use.

Arguments:
    None. Run with `python -m pytest tests/test_binning.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config import ExperimentConfig, SMOKE_MAX_TOKENS, SMOKE_REASONER_SPEC
from data.binning import DEFAULT_BINS, OPTION_A_BIN_COUNTS, doc_type_bin
from experiments.corpus import sample_questions_per_bin
from experiments.smoke import SMOKE_DOCUMENTS, select_smoke_questions, smoke_doc_bins
from schema import Question


def make_question(index: int, doc_id: str, doc_type: str) -> Question:
    return Question(
        id=f"q{index:03d}",
        doc_id=doc_id,
        question="question?",
        gold_answer="answer",
        answer_format="Str",
        doc_type=doc_type,
        evidence_pages=(0,),
        evidence_sources=("Pure-text (Plain-text)",),
        hop="single",
        is_unanswerable=False,
    )


def test_doc_type_bin_option_a_all_classes() -> None:
    assert doc_type_bin("Administration/Industry file") == "text_heavy"
    assert doc_type_bin("Academic paper") == "text_heavy"
    assert doc_type_bin("Research report / Introduction") == "text_heavy"
    assert doc_type_bin("Financial report") == "in_between"
    assert doc_type_bin("Guidebook") == "in_between"
    assert doc_type_bin("Tutorial/Workshop") == "in_between"
    assert doc_type_bin("Brochure") == "visual_heavy"
    assert doc_type_bin(" research report/introduction ") == "text_heavy"

    with pytest.raises(ValueError):
        doc_type_bin("Invoice")


def test_option_a_counts_match_dataset_census() -> None:
    assert OPTION_A_BIN_COUNTS["text_heavy"].questions == 578
    assert OPTION_A_BIN_COUNTS["text_heavy"].documents == 70
    assert OPTION_A_BIN_COUNTS["in_between"].questions == 412
    assert OPTION_A_BIN_COUNTS["in_between"].documents == 50
    assert OPTION_A_BIN_COUNTS["visual_heavy"].questions == 101
    assert OPTION_A_BIN_COUNTS["visual_heavy"].documents == 15


def test_smoke_set_is_non_empty_in_every_bin() -> None:
    questions = [
        make_question(index, document.doc_id, document.doc_type)
        for index, document in enumerate(SMOKE_DOCUMENTS)
    ]

    selected = select_smoke_questions(questions, require_all_docs=True)
    selected_bins = {doc_type_bin(question.doc_type) for question in selected}

    assert selected_bins == set(DEFAULT_BINS)
    assert set(smoke_doc_bins().values()) == set(DEFAULT_BINS)
    assert len(SMOKE_DOCUMENTS) == 7


def _bin_corpus() -> list[Question]:
    """Synthetic corpus: text_heavy/in_between over target, visual_heavy under it."""

    questions: list[Question] = []
    index = 0
    plan = {
        "Academic paper": 30,       # text_heavy: 30 docs x 5 Q = 150 (> 100)
        "Guidebook": 24,            # in_between: 24 docs x 5 Q = 120 (> 100)
        "Brochure": 3,              # visual_heavy: 3 docs x 5 Q = 15  (< 100)
    }
    for doc_type, n_docs in plan.items():
        for doc in range(n_docs):
            doc_id = f"{doc_type[:4]}-{doc:02d}"
            for _ in range(5):
                questions.append(make_question(index, doc_id, doc_type))
                index += 1
    return questions


def test_per_bin_sample_draws_whole_documents_to_target() -> None:
    questions = _bin_corpus()
    selected = sample_questions_per_bin(questions, 100, bins=DEFAULT_BINS, seed=0)

    per_bin: dict[str, list[Question]] = {b: [] for b in DEFAULT_BINS}
    for question in selected:
        per_bin[doc_type_bin(question.doc_type)].append(question)

    # Over-target bins land just past 100, always in whole 5-question documents.
    for bin_name in ("text_heavy", "in_between"):
        count = len(per_bin[bin_name])
        assert 100 <= count < 150
        assert count % 5 == 0
    # A bin below the target is kept whole.
    assert len(per_bin["visual_heavy"]) == 15

    # Document-level integrity: a chosen document keeps all its questions.
    kept_docs = {q.doc_id for q in selected}
    by_doc: dict[str, int] = {}
    for question in questions:
        by_doc[question.doc_id] = by_doc.get(question.doc_id, 0) + 1
    for doc_id in kept_docs:
        assert sum(1 for q in selected if q.doc_id == doc_id) == by_doc[doc_id]


def test_per_bin_sample_is_deterministic_and_seed_sensitive() -> None:
    questions = _bin_corpus()
    ids0a = [q.id for q in sample_questions_per_bin(questions, 100, bins=DEFAULT_BINS, seed=0)]
    ids0b = [q.id for q in sample_questions_per_bin(questions, 100, bins=DEFAULT_BINS, seed=0)]
    ids1 = [q.id for q in sample_questions_per_bin(questions, 100, bins=DEFAULT_BINS, seed=1)]

    assert ids0a == ids0b          # same seed -> identical subset
    assert set(ids0a) != set(ids1)  # a different seed draws different documents


def test_config_quantization_appends_spec_suffix() -> None:
    assert ExperimentConfig(smoke=False).reasoner_spec == "qwen3vl-8b-local"
    assert ExperimentConfig(smoke=False, quantization="4bit").reasoner_spec == "qwen3vl-8b-local-4bit"
    assert ExperimentConfig(smoke=False, quantization="8bit").reasoner_spec == "qwen3vl-8b-local-8bit"
    with pytest.raises(ValueError):
        ExperimentConfig(smoke=False, quantization="3bit")


def test_experiment_config_smoke_uses_root_relative_paths() -> None:
    config = ExperimentConfig(smoke=True)
    root = config.paths.root.resolve()

    for path in (
        config.paths.data_dir,
        config.paths.hf_home,
        config.paths.results_dir,
        config.paths.cache_dir,
        config.paths.env_dir,
    ):
        Path(path).resolve().relative_to(root)

    assert config.smoke
    assert config.reasoner_spec == SMOKE_REASONER_SPEC
    assert config.max_tokens == SMOKE_MAX_TOKENS
    assert config.bins == DEFAULT_BINS
    assert config.cost_metric == "latency_bs1"
    assert config.representations == ("T", "TL", "TLV", "V")
    assert config.sufficiency_margin == 3.0
