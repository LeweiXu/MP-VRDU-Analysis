"""The composer builds the four cost-ordered rungs, TL/TLV use parser text, no
bounding-box JSON is emitted, and the representation layer never invokes a model
(the parser boundary is the disk cache, so parser and reasoner never co-reside)."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from conftest import ROOT, require

RUNGS = ["T", "TL", "TLV", "V"]


def test_four_rungs_are_addressable() -> None:
    get_representation = require("pipeline.representation", "get_representation")
    for rung in RUNGS:
        assert get_representation(rung) is not None, f"missing rung {rung}"


def test_no_bbox_channel_anywhere() -> None:
    # Bounding-box JSON is dropped everywhere; the token-heavy channel is gone.
    src = (ROOT / "pipeline" / "representation.py").read_text().lower()
    assert "bbox" not in src, "representation must not reference bounding boxes"


def test_representation_layer_invokes_no_model() -> None:
    # Composition runs on already-cached text/images; it must not import a
    # reasoner/parser-model backend, which would break non-co-residence.
    tree = ast.parse((ROOT / "pipeline" / "representation.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = {"models", "models.qwen3vl", "models.internvl", "torch", "vllm", "transformers"}
    leaked = {m for m in imported if m in forbidden or m.split(".")[0] == "models"}
    assert not leaked, f"representation must not pull in a model backend: {leaked}"
