"""Shared test fixtures. All tests run offline against the synthetic fixture."""

import os

import pytest

from mpvrdu.data.synthetic import build_synthetic


@pytest.fixture
def chdir_tmp(tmp_path, monkeypatch):
    """Run a test inside a fresh tmp cwd so relative artifact dirs stay hermetic."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def synthetic_ds(tmp_path):
    """A small synthetic MMLongBench-Doc-shaped dataset on disk."""
    return build_synthetic(tmp_path / "syn")
