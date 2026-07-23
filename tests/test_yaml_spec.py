"""The flat spec format parses into runs and configs: task_name is a label, dataset
and parser cross-product into run_tags, quantization folds into reasoner_specs, the
retrieval benchmark enforces its fixed arms, and corpus.pool reaches the config."""

from __future__ import annotations

import pytest

from conftest import ROOT, require


def _flat(**over) -> dict:
    base = {
        "task_name": "G1_oracle_ladder",
        "run_tag": "g1",
        "corpus": {"pool": "answerable", "sampling": {"per_doc_type": 5, "seed": 0}},
        "retrieval_representation": ["oracle"],
        "reasoner_spec": ["qwen3vl-8b-local"],
        "reasoner_representations": ["T", "TL", "TLV", "V"],
    }
    base.update(over)
    return base


def _g2(**over) -> dict:
    base = {
        "task_name": "G2_retrieval",
        "run_tag": "g2",
        "retrieval_representation": ["T", "V"],
        "text_retrievers": ["bm25", "bge-m3", "qwen3-embedding"],
        "vision_retrievers": ["colmodernvbert", "colqwen2.5", "colqwen3"],
        "joints": "matched",
        "k_values": [1, 3],
        "joint_k_values": [1],
        "inference_text_retriever": "bge-m3",
        "inference_vision_retriever": "colqwen2.5",
        "inference_joint": True,
        "reasoner_representations": ["TLV", "V"],
    }
    base.update(over)
    return base


def test_flat_run_parses_to_one_spec() -> None:
    parse_specs = require("experiments.corpus.yaml_spec", "parse_specs")
    (spec,) = parse_specs(_flat())
    assert spec.task_name == "G1_oracle_ladder"
    assert spec.reasoner_representations == ("T", "TL", "TLV", "V")
    assert spec.retrieval_representation == ("oracle",)


def test_unknown_or_machine_key_rejected() -> None:
    parse_specs = require("experiments.corpus.yaml_spec", "parse_specs")
    SpecError = require("experiments.corpus.yaml_spec", "SpecError")
    with pytest.raises(SpecError):
        parse_specs(_flat(machine="supervisor"))


def test_no_saved_spec_declares_a_machine_field() -> None:
    offenders = [p.name for p in (ROOT / "ops" / "specs").glob("*.yaml") if "machine:" in p.read_text()]
    assert not offenders, f"specs must not mention machine: {offenders}"


def test_shipped_specs_load() -> None:
    load_yaml_specs = require("experiments.corpus.yaml_spec", "load_yaml_specs")
    config_from_spec = require("experiments.corpus.yaml_spec", "config_from_spec")
    for path in sorted((ROOT / "ops" / "specs").glob("*.yaml")):
        if path.name == "target_template.yaml":
            # The TARGET vocabulary: its [NEW] keys (page_set, corpus.hop) are
            # the subject of PIPELINE_EXTENSION_PLAN.md Phase B and do not load
            # yet. Drop this exclusion when page_set parsing lands.
            continue
        specs = load_yaml_specs(path)
        assert specs, f"{path.name} produced no runs"
        for spec in specs:
            config_from_spec(spec)


def test_reasoner_spec_list_drives_size_sweep() -> None:
    parse_specs = require("experiments.corpus.yaml_spec", "parse_specs")
    config_from_spec = require("experiments.corpus.yaml_spec", "config_from_spec")
    get_task = require("experiments.registry", "get_task")
    sizes = ["qwen3vl-2b-local", "qwen3vl-4b-local", "qwen3vl-8b-local"]
    (spec,) = parse_specs(_flat(reasoner_spec=sizes))
    assert spec.reasoner_specs == tuple(sizes)
    config = config_from_spec(spec)
    assert tuple(get_task("G1_oracle_ladder").model_specs(config)) == tuple(sizes)


def test_quantization_folds_into_reasoner_specs() -> None:
    parse_specs = require("experiments.corpus.yaml_spec", "parse_specs")
    (spec,) = parse_specs(_flat(reasoner_spec=["qwen3vl-8b-local"], quantization=["bf16", "4bit"]))
    # bf16 stays bare; the quantized variant gets the suffix, so both key distinctly.
    assert spec.reasoner_specs == ("qwen3vl-8b-local", "qwen3vl-8b-local-4bit")


def test_dataset_and_parser_cross_product_run_tags() -> None:
    parse_specs = require("experiments.corpus.yaml_spec", "parse_specs")
    specs = parse_specs(_flat(dataset=["mmlongbench", "longdocurl"], parser=["paddleocrvl", "mineru"]))
    assert {s.run_tag for s in specs} == {
        "g1-mmlongbench-paddleocrvl", "g1-mmlongbench-mineru",
        "g1-longdocurl-paddleocrvl", "g1-longdocurl-mineru",
    }


def test_multi_run_rejects_duplicate_run_tags() -> None:
    parse_specs = require("experiments.corpus.yaml_spec", "parse_specs")
    SpecError = require("experiments.corpus.yaml_spec", "SpecError")
    with pytest.raises(SpecError):
        parse_specs({"runs": [_flat(run_tag="dup"), _g2(run_tag="dup")]})


def test_benchmark_requires_bge_m3_and_colqwen25() -> None:
    parse_specs = require("experiments.corpus.yaml_spec", "parse_specs")
    SpecError = require("experiments.corpus.yaml_spec", "SpecError")
    parse_specs(_g2())  # ok: both fixed arms present
    with pytest.raises(SpecError):
        parse_specs(_g2(text_retrievers=["bm25"], inference_text_retriever="bm25"))
    with pytest.raises(SpecError):
        parse_specs(_g2(vision_retrievers=["colmodernvbert"], inference_vision_retriever="colmodernvbert"))


def test_inference_pick_must_be_a_benchmarked_method() -> None:
    parse_specs = require("experiments.corpus.yaml_spec", "parse_specs")
    SpecError = require("experiments.corpus.yaml_spec", "SpecError")
    with pytest.raises(SpecError):
        parse_specs(_g2(text_retrievers=["bge-m3"], inference_text_retriever="qwen3-embedding"))


def test_config_maps_pool_and_retrieval_representation() -> None:
    parse_specs = require("experiments.corpus.yaml_spec", "parse_specs")
    config_from_spec = require("experiments.corpus.yaml_spec", "config_from_spec")
    (spec,) = parse_specs(_flat(corpus={"pool": "unanswerable", "sampling": "full"},
                                retrieval_representation=["T"]))
    config = config_from_spec(spec)
    assert config.pool == "unanswerable"
    assert config.retrieval_representation == ("T",)


def test_config_maps_g2_fields() -> None:
    parse_specs = require("experiments.corpus.yaml_spec", "parse_specs")
    config_from_spec = require("experiments.corpus.yaml_spec", "config_from_spec")
    (spec,) = parse_specs(_g2())
    config = config_from_spec(spec)
    assert config.text_retrievers == ("bm25", "bge-m3", "qwen3-embedding")
    assert config.inference_text_retriever == "bge-m3"
    assert config.inference_joint is True
    assert config.k_values == (1, 3)
    assert config.joint_k_values == (1,)
