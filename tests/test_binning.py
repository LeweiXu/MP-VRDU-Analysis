"""Stage M1 tests for Option-A binning, smoke corpus, and config knobs."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import ExperimentConfig, SMOKE_MAX_TOKENS, SMOKE_REASONER_SPEC
from data.binning import DEFAULT_BINS, OPTION_A_BIN_COUNTS, doc_type_bin
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
