"""Every spine/ops module has a concise, present-tense docstring with no plan-talk
(no roadmap, "will change", legacy/v4/pivot, or RQ/table numbers)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from conftest import ROOT

SPINE_DIRS = [
    "data", "tools", "retrievers", "models", "pipeline", "scoring",
    "experiments", "reporting",
]
# ops entry points are v4-authored; ops/scripts and ops/kaya are copied tooling
# reworked in Phase 4, so they are not held to the docstring rule yet.
ROOT_FILES = ["config.py", "schema.py",
              "ops/__init__.py", "ops/generate.py", "ops/judge.py", "ops/build.py"]

# Plan-talk the docstring rule forbids.
FORBIDDEN = re.compile(
    r"\bv4\b|\bv3\b|\blegacy\b|\bpivot\b|\broadmap\b|will change|should update|"
    r"\bRQ\d|\bTable\s*\d|\bT[1-9]\b",
    re.IGNORECASE,
)


def _modules() -> list[Path]:
    paths: list[Path] = []
    for name in ROOT_FILES:
        paths.append(ROOT / name)
    for d in SPINE_DIRS:
        paths.extend(p for p in (ROOT / d).rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(paths)


@pytest.mark.parametrize("path", _modules(), ids=lambda p: str(p.relative_to(ROOT)))
def test_module_has_clean_docstring(path: Path) -> None:
    doc = ast.get_docstring(ast.parse(path.read_text())) or ""
    rel = path.relative_to(ROOT)
    assert doc.strip(), f"{rel}: missing module docstring"
    bad = FORBIDDEN.search(doc)
    assert not bad, f"{rel}: docstring contains plan-talk: {bad.group(0)!r}"
    # Concise: 1-3 sentences.
    assert len(doc.split()) <= 60, f"{rel}: docstring too long ({len(doc.split())} words)"
