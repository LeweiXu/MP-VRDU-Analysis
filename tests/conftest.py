"""Shared fixtures and helpers for the v4 test suite.

These tests are written before the v4 spine is implemented: they are the
executable spec Phase 4 must satisfy, so most are red against the scaffolded
stubs. `require()` turns a not-yet-implemented symbol into a clear, single-line
failure instead of a collection-time ImportError, keeping the whole suite
collectable and uniformly red.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "v3_results"


def require(module_path: str, attr: str):
    """Return `module_path.attr`, or fail the test if the stub lacks it yet.

    The module itself must import (it exists as a scaffolded stub); only the
    attribute is allowed to be missing, which marks unfinished v4 work as a red
    test rather than an error.
    """
    module = importlib.import_module(module_path)
    obj = getattr(module, attr, None)
    if obj is None:
        pytest.fail(f"{module_path}.{attr} not implemented yet (v4 stub)")
    return obj


def read_jsonl(path: Path) -> list[dict]:
    """Parse a jsonl file into a list of dicts (used to read the v3 fixtures)."""
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    assert FIXTURES.is_dir(), f"missing Phase-0 fixtures at {FIXTURES}"
    return FIXTURES


@pytest.fixture(scope="session")
def g1_predictions() -> list[dict]:
    """Real v3-shaped prediction rows (shape reference; values not comparable)."""
    return read_jsonl(FIXTURES / "bf16-lowres" / "full" / "G1_sufficiency" / "predictions.jsonl")


@pytest.fixture(scope="session")
def g1_results() -> list[dict]:
    """Real v3-shaped judged rows."""
    return read_jsonl(FIXTURES / "bf16-lowres" / "full" / "G1_sufficiency" / "results.jsonl")


@pytest.fixture(scope="session")
def g5_retrieval() -> list[dict]:
    """Real v3-shaped retrieval side-artifact rows."""
    return read_jsonl(FIXTURES / "bf16-lowres" / "full" / "G5_retrieval" / "retrieval.jsonl")


@pytest.fixture(scope="session")
def g6_classifier() -> list[dict]:
    """Real v3-shaped classifier side-artifact rows."""
    return read_jsonl(FIXTURES / "bf16-lowres" / "full" / "G6_classifier" / "classifier.jsonl")
