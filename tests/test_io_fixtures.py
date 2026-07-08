"""I/O plumbing against the real v3-shaped fixtures: the jsonl reader parses real
rows, the build step groups them at the right cardinality, and the side-artifact
readers parse the retrieval and classifier artifacts. Shape only, values are not
comparable to v4."""

from __future__ import annotations

import pytest

from conftest import require


# ---- fixtures are wired (these pass now: they only touch the captured files) ----

def test_prediction_rows_parse_with_expected_keys(g1_predictions) -> None:
    assert g1_predictions, "no prediction rows parsed"
    row = g1_predictions[0]
    for key in ("prediction_key", "question_id", "doc_id", "representation", "model_spec"):
        assert key in row, f"prediction row missing {key}"


def test_result_rows_carry_judge_output(g1_results) -> None:
    row = g1_results[0]
    for key in ("score", "correct", "judge_spec", "question_id"):
        assert key in row


def test_retrieval_artifact_shape(g5_retrieval) -> None:
    row = g5_retrieval[0]
    for key in ("question_id", "retriever", "modality", "k", "retrieved_pages", "gold_pages",
                "precision", "recall", "f1"):
        assert key in row


def test_classifier_artifact_shape(g6_classifier) -> None:
    row = g6_classifier[0]
    for key in ("classifier", "doc_id", "predicted_bin", "gold_bin"):
        assert key in row


# ---- v4 readers/build parse the same shapes (red until Phase 4) ----

def test_v4_reader_parses_fixture_rows(fixtures_dir) -> None:
    read_rows = require("experiments.engine.driver", "read_rows")
    path = fixtures_dir / "bf16-lowres" / "full" / "G1_sufficiency" / "predictions.jsonl"
    rows = list(read_rows(path))
    assert len(rows) > 1000, "reader should parse every row"


def test_build_groups_rows_at_cell_cardinality(g1_results) -> None:
    group_rows = require("reporting.build", "group_rows")
    groups = group_rows(g1_results)
    # One group per prediction identity; never more groups than rows.
    assert 0 < len(groups) <= len(g1_results)
