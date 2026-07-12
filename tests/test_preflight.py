"""The login-node preflight resolves specs, weights, and --gres before a submit,
and `kaya submit` extracts the spec to gate on it."""
from __future__ import annotations

import pytest

from ops.kaya.kaya import spec_arg
from ops.scripts.preflight import (
    Report,
    _parse_gres,
    _reasoner_model_id,
    _retriever_model_id,
)


def test_spec_arg_extracts_from_forwarded():
    assert spec_arg(["--spec", "ops/specs/x.yaml"]) == "ops/specs/x.yaml"
    assert spec_arg(["--spec=ops/specs/y.yaml", "--limit", "2"]) == "ops/specs/y.yaml"
    assert spec_arg(["--task", "all"]) is None
    assert spec_arg([]) is None


def test_parse_gres():
    assert _parse_gres("gpu:v100:2") == (2, "v100")
    assert _parse_gres("gpu:1") == (1, None)
    assert _parse_gres(None) == (None, None)
    assert _parse_gres("garbage") == (None, None)


def test_reasoner_model_id_handles_families_and_quant():
    assert _reasoner_model_id("qwen3vl-8b-local") == "Qwen/Qwen3-VL-8B-Instruct"
    # a quantization suffix resolves to the same base checkpoint
    assert _reasoner_model_id("qwen3vl-2b-local-4bit") == "Qwen/Qwen3-VL-2B-Instruct"
    assert _reasoner_model_id("internvl3-8b-local") == "OpenGVLab/InternVL3-8B"
    assert _reasoner_model_id("stub") is None


def test_retriever_model_id():
    assert _retriever_model_id("bge-m3", "text") == "BAAI/bge-m3"
    assert _retriever_model_id("qwen3-embedding", "text") == "Qwen/Qwen3-Embedding-4B"
    assert _retriever_model_id("colqwen2.5", "vision") == "vidore/colqwen2.5-v0.2"
    # bm25 / none / unset carry no weights
    assert _retriever_model_id("bm25", "text") is None
    assert _retriever_model_id("none", "vision") is None


def test_report_fails_only_on_hard_failures():
    report = Report()
    report.ok("a", "fine")
    report.warn("b", "meh")
    assert report.failed is False
    report.fail("c", "broken")
    assert report.failed is True


def test_reasoner_model_id_rejects_unknown_family():
    with pytest.raises(ValueError):
        _reasoner_model_id("llama-7b-local")
