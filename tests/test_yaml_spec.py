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


def test_shipped_specs_load() -> None:
    # Every checked-in spec parses (flat, multi-run, or the nested base/sweeps form)
    # and builds a config per expanded run.
    load_yaml_specs = require("experiments.corpus.yaml_spec", "load_yaml_specs")
    config_from_spec = require("experiments.corpus.yaml_spec", "config_from_spec")
    for path in sorted((ROOT / "ops" / "specs").glob("*.yaml")):
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


# -- the nested base/sweeps + retrieval/inference expander --------------------

def _g1_nested() -> dict:
    return {
        "base": {"corpus": {"sampling": {"per_doc_type": 50, "seed": 0}},
                 "parser": "paddleocrvl", "dataset": "mmlongbench"},
        "runs": [
            {"task": "G1_oracle_ladder", "run_tag": "g1",
             "base": {"reasoner_spec": "qwen3vl-8b-local", "quantization": "bf16",
                      "visual_resolution": "med", "representations": ["T", "TL", "TLV", "V"]},
             "sweeps": {
                 "size": {"reasoner_spec": ["qwen3vl-2b-local", "qwen3vl-8b-local"]},
                 "quantization": {"quantization": ["bf16", "4bit"]},
                 "resolution": {"visual_resolution": ["low", "med", "high"],
                                "representations": ["TLV", "V"]},
                 "parser": {"parser": ["paddleocrvl", "mineru"], "representations": ["TL", "TLV"]},
                 "dataset": {"dataset": ["mmlongbench", "longdocurl"]},
             }},
        ],
    }


def test_expander_g1_sweeps_run_tags_and_axes() -> None:
    parse_specs = require("experiments.corpus.yaml_spec", "parse_specs")
    specs = {s.run_tag: s for s in parse_specs(_g1_nested())}
    assert set(specs) == {
        "g1", "g1-size", "g1-quantization", "g1-resolution",
        "g1-parser-paddleocrvl", "g1-parser-mineru",
        "g1-dataset-mmlongbench", "g1-dataset-longdocurl",
    }
    # bf16 base -> unquantized; file+task base merged onto every spec.
    assert specs["g1"].quantization is None
    assert specs["g1"].reasoner_spec == "qwen3vl-8b-local"
    assert specs["g1"].dataset == "mmlongbench"
    # size sweep -> a reasoner_specs list under one run_tag (axis is in the cache key).
    assert specs["g1-size"].reasoner_specs == ("qwen3vl-2b-local", "qwen3vl-8b-local")
    # quant sweep -> reasoner_specs carry the -4bit suffix, bf16 stays bare.
    assert specs["g1-quantization"].reasoner_specs == ("qwen3vl-8b-local", "qwen3vl-8b-local-4bit")
    assert specs["g1-quantization"].quantization is None
    # resolution sweep -> a visual_resolutions list + the coupled reps override.
    assert specs["g1-resolution"].visual_resolutions == ("low", "med", "high")
    assert specs["g1-resolution"].representations == ("TLV", "V")
    # parser sweep -> one run_tag PER value (parser is not in the cache key).
    assert specs["g1-parser-mineru"].parser == "mineru"
    assert specs["g1-parser-mineru"].representations == ("TL", "TLV")
    assert specs["g1-dataset-longdocurl"].dataset == "longdocurl"


def test_expander_collapses_single_value_sweep() -> None:
    parse_specs = require("experiments.corpus.yaml_spec", "parse_specs")
    raw = _g1_nested()
    # A size sweep of just the base value contributes nothing beyond the base run.
    raw["runs"][0]["sweeps"] = {"size": {"reasoner_spec": ["qwen3vl-8b-local"]}}
    tags = {s.run_tag for s in parse_specs(raw)}
    assert tags == {"g1"}


def test_expander_g3_prompt_sweep_folds_into_single_run() -> None:
    parse_specs = require("experiments.corpus.yaml_spec", "parse_specs")
    raw = {"runs": [
        {"task": "G3_hallucination", "run_tag": "g3",
         "base": {"reasoner_spec": "qwen3vl-8b-local", "corpus": {"sampling": "full"}},
         "sweeps": {"prompt": {"prompt_mode": ["none", "generic", "targeted"]}}},
    ]}
    specs = parse_specs(raw)
    assert [s.run_tag for s in specs] == ["g3"]  # no extra run_tag for the prompt sweep
    assert specs[0].prompt_modes == ("none", "generic", "targeted")


def test_expander_g2_retrieval_inference_and_subset_check() -> None:
    parse_specs = require("experiments.corpus.yaml_spec", "parse_specs")
    SpecError = require("experiments.corpus.yaml_spec", "SpecError")
    good = {"runs": [
        {"task": "G2_retrieval", "run_tag": "g2",
         "retrieval": {"text_retrievers": ["bm25", "bge-m3"], "vision_retrievers": ["colqwen2.5"],
                       "joints": "matched", "k_values": [1, 3], "joint_k_values": [1]},
         "inference": {"reasoner_spec": "qwen3vl-8b-local", "text_retriever": "bm25",
                       "vision_retriever": "colqwen2.5", "joint": False,
                       "representations": ["TLV", "V"], "k_values": [1, 3]}},
    ]}
    (spec,) = parse_specs(good)
    assert spec.text_retrievers == ("bm25", "bge-m3")
    assert spec.inference_text_retriever == "bm25"
    assert spec.inference_joint is False
    assert spec.inference_representations == ("TLV", "V")

    bad = {"runs": [
        {"task": "G2_retrieval", "run_tag": "g2",
         "retrieval": {"text_retrievers": ["bm25"], "vision_retrievers": ["colqwen2.5"]},
         "inference": {"text_retriever": "bge-m3", "vision_retriever": "colqwen2.5"}},
    ]}
    with pytest.raises(SpecError):
        parse_specs(bad)


def test_expander_config_from_spec_maps_g2_fields() -> None:
    parse_specs = require("experiments.corpus.yaml_spec", "parse_specs")
    config_from_spec = require("experiments.corpus.yaml_spec", "config_from_spec")
    raw = {"runs": [
        {"task": "G2_retrieval", "run_tag": "g2",
         "retrieval": {"text_retrievers": ["bm25", "bge-m3"], "vision_retrievers": ["colqwen2.5"],
                       "joints": "matched", "k_values": [1, 3], "joint_k_values": [1]},
         "inference": {"text_retriever": "bm25", "vision_retriever": "colqwen2.5",
                       "joint": True, "representations": ["V"], "k_values": [1, 3]}},
    ]}
    (spec,) = parse_specs(raw)
    config = config_from_spec(spec)
    assert config.text_retrievers == ("bm25", "bge-m3")
    assert config.inference_text_retriever == "bm25"
    assert config.inference_representations == ("V",)
    assert config.joint_k_values == (1,)
