"""Cell keying is machine-independent: identical inputs produce identical SHA-256
keys under simulated different environments (no device-count / hostname /
torch.cuda leak)."""

from __future__ import annotations

import pytest

from conftest import require

IDENTITY = {
    "question_id": "q1",
    "doc_id": "d1",
    "condition": "oracle",
    "representation": "TLV",
    "model_spec": "qwen3vl-8b",
    "page_indices": (0, 2, 5),
}


def test_key_is_sha256_hex() -> None:
    key_fn = require("experiments.engine.paths", "prediction_key")
    key = key_fn(**IDENTITY)
    assert isinstance(key, str) and len(key) == 64
    int(key, 16)  # hex


def test_key_stable_across_simulated_environments(monkeypatch) -> None:
    key_fn = require("experiments.engine.paths", "prediction_key")
    baseline = key_fn(**IDENTITY)

    # Simulate a different machine: hostname + a torch cuda device count. Nothing
    # machine-dependent may enter the key.
    import socket

    monkeypatch.setattr(socket, "gethostname", lambda: "supervisor-h100")
    try:
        import torch  # noqa: F401
        monkeypatch.setattr("torch.cuda.device_count", lambda: 1, raising=False)
    except Exception:
        pass

    assert key_fn(**IDENTITY) == baseline, "key must not depend on the environment"


def test_key_changes_with_representation() -> None:
    key_fn = require("experiments.engine.paths", "prediction_key")
    other = dict(IDENTITY, representation="V")
    assert key_fn(**IDENTITY) != key_fn(**other)
