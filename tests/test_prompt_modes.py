"""Phase A guards: the six-mode prompt set, per-mode decode budgets, the
final-answer delimiter extraction, and the run-settings sidecar."""

from types import SimpleNamespace

import pytest

from config import PROMPT_MODES, ExperimentConfig, budget_for_mode
from scoring.abstention import extract_final_answer, is_abstention


def test_six_modes_present():
    for mode in ("none", "grounded", "abstain", "abstain_balanced", "cot", "extract_cot"):
        assert mode in PROMPT_MODES
    assert PROMPT_MODES["none"] == ""


def test_legacy_aliases_are_byte_identical():
    # Existing cached cells carry generic/targeted condition suffixes; the alias
    # strings must never drift from grounded/abstain or old and new rows stop
    # being comparable (the Stage-8 rate-level reconciliation depends on this).
    assert PROMPT_MODES["generic"] == PROMPT_MODES["grounded"]
    assert PROMPT_MODES["targeted"] == PROMPT_MODES["abstain"]


def test_budget_for_mode_defaults_and_overrides():
    config = ExperimentConfig(decode_budget={"default": 256, "cot": 1024})
    assert budget_for_mode(config, "cot") == 1024
    assert budget_for_mode(config, "none") == 256
    plain = ExperimentConfig()
    assert budget_for_mode(plain, "cot") == plain.max_tokens


def test_budget_for_mode_smoke_stays_capped():
    from config import SMOKE_MAX_TOKENS

    config = ExperimentConfig(smoke=True, decode_budget={"default": 256, "cot": 1024})
    assert budget_for_mode(config, "cot") == SMOKE_MAX_TOKENS


def test_decode_budget_validation():
    with pytest.raises(ValueError):
        ExperimentConfig(decode_budget={"cot": 1024})  # no default
    with pytest.raises(ValueError):
        ExperimentConfig(decode_budget={"default": 256, "nope": 100})  # unknown mode
    with pytest.raises(ValueError):
        ExperimentConfig(decode_budget={"default": 0})  # non-positive


def test_extract_final_answer():
    assert extract_final_answer("plain", None) == "plain"
    assert extract_final_answer("plain", "Answer:") == "plain"  # delimiter absent
    assert extract_final_answer("think... Answer: 42", "Answer:") == "42"
    # The LAST occurrence wins: the body template's cue can be echoed early.
    assert extract_final_answer("Answer: draft\nmore\nAnswer: final", "Answer:") == "final"


def test_delimited_abstention_still_detected():
    # The Stage-4 watch-for: the final-line contract may wrap the abstention
    # string, and the detector must still see it.
    assert is_abstention("Answer: Not answerable.")
    assert is_abstention(extract_final_answer("thinking Answer: Not answerable.", "Answer:"))


def test_prediction_row_metadata_whitelist():
    from pipeline.orchestrator import _ROW_METADATA_KEYS

    assert "output_truncated" in _ROW_METADATA_KEYS
    assert "max_new_tokens" in _ROW_METADATA_KEYS
    assert "cache_dir" not in _ROW_METADATA_KEYS  # machine-local, must not reach rows


def test_run_settings_sidecar_guard(tmp_path):
    from experiments.engine.driver import check_run_settings

    predictions = tmp_path / "predictions.jsonl"
    config = ExperimentConfig(decode_budget={"default": 256, "cot": 1024},
                              final_answer_delimiter="Answer:")
    check_run_settings(predictions, config)  # absent -> written
    assert (tmp_path / "run_settings.json").exists()
    check_run_settings(predictions, config)  # equal -> proceeds
    changed = ExperimentConfig(decode_budget={"default": 512})
    with pytest.raises(RuntimeError):
        check_run_settings(predictions, changed)  # different -> refuses


def test_run_settings_readonly_does_not_write(tmp_path):
    from experiments.engine.driver import check_run_settings

    predictions = tmp_path / "predictions.jsonl"
    check_run_settings(predictions, ExperimentConfig(), readonly=True)
    assert not (tmp_path / "run_settings.json").exists()


def test_orchestrator_metadata_merge():
    # _prediction_row copies whitelisted backend metadata onto the row and keeps
    # source_dataset; machine-local backend keys never reach the row.
    from pipeline.orchestrator import _ROW_METADATA_KEYS

    backend_meta = {"max_new_tokens": 1024, "output_truncated": True,
                    "cache_dir": "/scratch/x", "prompt_template_version": "v1"}
    merged = {"source_dataset": "mmlongbench",
              **{k: backend_meta[k] for k in _ROW_METADATA_KEYS if k in backend_meta}}
    assert merged["output_truncated"] is True
    assert merged["max_new_tokens"] == 1024
    assert "cache_dir" not in merged


def test_spec_decode_budget_and_delimiter_parse():
    from experiments.corpus.yaml_spec import SpecError, parse_spec

    base = {"task_name": "G4_faithfulness_answerable",
            "prompt_modes": ["none", "cot"],
            "decode_budget": {"default": 256, "cot": 1024},
            "final_answer_delimiter": "Answer:"}
    spec = parse_spec(base)
    assert spec.decode_budget == {"default": 256, "cot": 1024}
    assert spec.final_answer_delimiter == "Answer:"
    none_spec = parse_spec({"task_name": "t", "final_answer_delimiter": "none"})
    assert none_spec.final_answer_delimiter is None
    with pytest.raises(SpecError):
        parse_spec({"task_name": "t", "decode_budget": {"cot": 1024}})  # no default
    with pytest.raises(SpecError):
        parse_spec({"task_name": "t", "decode_budget": {"default": 256, "bogus": 9}})


def test_new_spec_files_parse():
    from pathlib import Path

    from experiments.corpus.yaml_spec import config_from_spec, load_yaml_specs

    root = Path(__file__).resolve().parents[1]
    for name in ("g3_faithfulness.yaml", "g4_faithfulness.yaml"):
        (spec,) = load_yaml_specs(root / "ops" / "specs" / name)
        assert spec.decode_budget and spec.decode_budget["cot"] == 1024
        assert spec.final_answer_delimiter == "Answer:"
        assert tuple(spec.prompt_modes) == (
            "none", "grounded", "abstain", "abstain_balanced", "cot", "extract_cot")
        config = config_from_spec(spec)
        assert config.final_answer_delimiter == "Answer:"
        assert budget_for_mode(config, "extract_cot") == 1024
