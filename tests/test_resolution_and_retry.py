"""Resolution is a per-cell key axis, --failed-only retries only failed cells, and
G3 sweeps all three prompt conditions (none / generic / targeted)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from conftest import require


def test_prediction_key_includes_resolution():
    pk = require("experiments.engine.paths", "prediction_key")
    ident = dict(question_id="q", doc_id="d", condition="oracle", representation="V",
                 model_spec="m", page_indices=(0,))
    assert pk(**ident) != pk(**ident, visual_resolution="high")
    assert pk(**ident, visual_resolution="low") != pk(**ident, visual_resolution="high")


def test_result_key_includes_resolution():
    rk = require("experiments.engine.paths", "result_key")
    ident = dict(question_id="q", doc_id="d", condition="oracle", representation="V",
                 model_spec="m", page_indices=(0,), judge_spec="stub")
    assert rk(**ident, visual_resolution="low") != rk(**ident, visual_resolution="high")


def test_config_validates_visual_resolutions():
    ExperimentConfig = require("config", "ExperimentConfig")
    assert ExperimentConfig(visual_resolutions=("low", "high")).visual_resolutions == ("low", "high")
    with pytest.raises(ValueError):
        ExperimentConfig(visual_resolutions=("bogus",))


def test_prepare_failed_only_drops_failed_rows_and_returns_resolution_identity(tmp_path):
    prepare = require("experiments.engine.driver", "_prepare_failed_only")
    path = tmp_path / "results.jsonl"
    rows = [
        {"status": "ok", "question_id": "q1", "doc_id": "d", "condition": "oracle",
         "representation": "V", "model_spec": "m", "visual_resolution": "low"},
        {"status": "error", "question_id": "q1", "doc_id": "d", "condition": "oracle",
         "representation": "V", "model_spec": "m", "visual_resolution": "high", "skipped_reason": "x"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    failed = prepare(path)
    assert failed == {("q1", "d", "oracle", "V", "m", "high")}
    remaining = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert len(remaining) == 1 and remaining[0]["status"] == "ok"


def test_cell_identity_includes_resolution():
    identity = require("experiments.engine.driver", "_cell_identity")
    cell = SimpleNamespace(question=SimpleNamespace(id="q1", doc_id="d"),
                           conditioner=SimpleNamespace(name="oracle"), representation="V")
    assert identity(cell, "m", "high") == ("q1", "d", "oracle", "V", "m", "high")


def test_merge_failed_only_upgrades_in_place():
    merge = require("experiments.engine.driver", "merge_failed_only")
    existing = [{"prediction_key": "a", "status": "ok"}, {"prediction_key": "b", "status": "error"}]
    reruns = [{"prediction_key": "b", "status": "ok"}]
    merged = merge(existing, reruns)
    assert [m["status"] for m in merged] == ["ok", "ok"]


def test_g3_prompt_modes_sweep_all_six():
    # The faithfulness sweeps need the unprompted 'none' baseline alongside the
    # five composed mechanisms (grounding, abstention escape, balanced escape,
    # CoT, extraction+CoT).
    modes = require("config", "G3_PROMPT_MODES")
    assert tuple(modes) == ("none", "grounded", "abstain", "abstain_balanced", "cot", "extract_cot")
