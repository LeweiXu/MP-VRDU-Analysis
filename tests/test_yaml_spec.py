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
