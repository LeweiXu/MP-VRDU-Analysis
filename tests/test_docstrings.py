"""Test repository-wide module documentation requirements.

Purpose:
    Enforces the implementation-plan rule that every Python file starts with a
    useful module docstring documenting purpose and arguments.

Test role:
    Catches new files whose top-level documentation is missing or too terse
    before they become part of later staged work.

Arguments:
    None. Run with `python -m pytest tests/test_docstrings.py`.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".cache", ".data", "envs", "results", "logs"}


def repo_python_files() -> list[Path]:
    """Return importable/source Python files, excluding generated artifact trees."""

    return sorted(
        path
        for path in ROOT.rglob("*.py")
        if not any(part in SKIP_PARTS for part in path.relative_to(ROOT).parts)
    )


def test_python_files_have_comprehensive_module_docstrings() -> None:
    missing: list[str] = []
    incomplete: list[str] = []

    for path in repo_python_files():
        module = ast.parse(path.read_text())
        docstring = ast.get_docstring(module) or ""
        rel = str(path.relative_to(ROOT))
        if not docstring:
            missing.append(rel)
            continue
        required_sections = ("Purpose:", "Arguments:")
        if any(section not in docstring for section in required_sections) or len(docstring.split()) < 25:
            incomplete.append(rel)

    assert not missing
    assert not incomplete
