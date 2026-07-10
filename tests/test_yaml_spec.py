"""YAML spec loading resolves a task + cell grid + corpus scope, and the machine
split does not exist as a concept (no `machine:` field anywhere)."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import ROOT, require

SPEC = {
    "task": "G1_oracle_ladder",
    "representations": ["T", "TL", "TLV", "V"],
    "corpus": {"sampling": {"per_bin": 30, "seed": 0}},
}


def test_spec_resolves_task_and_grid() -> None:
    parse_spec = require("experiments.corpus.yaml_spec", "parse_spec")
    spec = parse_spec(SPEC)
    assert getattr(spec, "task", None) == "G1_oracle_ladder"
    assert tuple(getattr(spec, "representations", ())) == ("T", "TL", "TLV", "V")


def test_spec_has_no_machine_concept() -> None:
    parse_spec = require("experiments.corpus.yaml_spec", "parse_spec")
    spec = parse_spec(SPEC)
    assert not hasattr(spec, "machine"), "specs must not carry a machine field"
    # A spec that declares a machine field is rejected outright.
    with pytest.raises(Exception):
        parse_spec(dict(SPEC, machine="supervisor"))


def test_no_saved_spec_declares_a_machine_field() -> None:
    specs_dir = ROOT / "ops" / "specs"
    offenders = [p.name for p in specs_dir.glob("*.yaml")
                 if "machine:" in p.read_text()]
    assert not offenders, f"specs must not mention machine: {offenders}"


def test_reasoner_specs_drives_a_size_sweep() -> None:
    # A spec with reasoner_specs makes G1 generate one pass per spec.
    parse_spec = require("experiments.corpus.yaml_spec", "parse_spec")
    config_from_spec = require("experiments.corpus.yaml_spec", "config_from_spec")
    get_task = require("experiments.registry", "get_task")
    sizes = ["qwen3vl-2b-local", "qwen3vl-4b-local", "qwen3vl-8b-local", "qwen3vl-32b-local"]
    spec = parse_spec({"task": "G1_oracle_ladder", "reasoner_specs": sizes})
    config = config_from_spec(spec)
    assert tuple(get_task("G1_oracle_ladder").model_specs(config)) == tuple(sizes)
    # Absent, the single reasoner_spec is used (behaviour unchanged).
    single = config_from_spec(parse_spec({"task": "G1_oracle_ladder"}))
    assert len(get_task("G1_oracle_ladder").model_specs(single)) == 1


# The per-sweep design mock-up: nested base/sweeps/retrieval/inference that the
# flat parser does not yet expand. It ships as a commented reference; the expander
# that makes it parse (and this exclusion) lands later.
_UNWIRED_SPECS = {"target_architecture.yaml"}


def test_shipped_specs_load() -> None:
    # Every checked-in spec parses (flat or multi-run) and builds a config per run.
    load_yaml_specs = require("experiments.corpus.yaml_spec", "load_yaml_specs")
    config_from_spec = require("experiments.corpus.yaml_spec", "config_from_spec")
    for path in sorted((ROOT / "ops" / "specs").glob("*.yaml")):
        if path.name in _UNWIRED_SPECS:
            continue
        specs = load_yaml_specs(path)
        assert specs, f"{path.name} produced no runs"
        for spec in specs:
            config_from_spec(spec)


def test_multi_run_merges_base_and_isolates_run_tags() -> None:
    parse_specs = require("experiments.corpus.yaml_spec", "parse_specs")
    raw = {
        "base": {"corpus": {"sampling": "full"}, "parser": "paddleocrvl"},
        "runs": [
            {"task": "G1_oracle_ladder", "run_tag": "g1-size",
             "reasoner_specs": ["qwen3vl-2b-local", "qwen3vl-8b-local"]},
            {"task": "G1_oracle_ladder", "run_tag": "g1-res-low",
             "reasoner_spec": "qwen3vl-8b-local", "visual_resolution": "low",
             "representations": ["TLV", "V"]},
        ],
    }
    specs = parse_specs(raw)
    assert [s.run_tag for s in specs] == ["g1-size", "g1-res-low"]
    # base is inherited, per-run keys win.
    assert all(s.parser == "paddleocrvl" for s in specs)
    assert specs[0].reasoner_specs == ("qwen3vl-2b-local", "qwen3vl-8b-local")
    assert specs[1].visual_resolution == "low"
    assert specs[1].representations == ("TLV", "V")


def test_multi_run_requires_unique_present_run_tags() -> None:
    parse_specs = require("experiments.corpus.yaml_spec", "parse_specs")
    with pytest.raises(Exception):  # missing run_tag
        parse_specs({"runs": [{"task": "G1_oracle_ladder"}]})
    with pytest.raises(Exception):  # duplicate run_tag
        parse_specs({"runs": [
            {"task": "G1_oracle_ladder", "run_tag": "dup"},
            {"task": "G2_retrieval", "run_tag": "dup"},
        ]})


def test_flat_spec_is_a_single_run() -> None:
    parse_specs = require("experiments.corpus.yaml_spec", "parse_specs")
    specs = parse_specs({"task": "G1_oracle_ladder"})
    assert len(specs) == 1 and specs[0].task == "G1_oracle_ladder"
