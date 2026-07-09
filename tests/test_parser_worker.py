"""Tests for strict parser-backend selection."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import tools.parser_worker as worker


def test_paddleocrvl_always_uses_v1_full_model(monkeypatch) -> None:
    calls = []

    class FakePaddleOCRVL:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def predict(self, image_path):
            calls.append(("predict", image_path))
            return [{"markdown": {"markdown_texts": f"text from {image_path}"}}]

    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        SimpleNamespace(PaddleOCRVL=FakePaddleOCRVL),
    )
    monkeypatch.setattr(worker, "_PADDLE_VL", None)

    assert worker._paddleocrvl("page-1.png", "ignored") == "text from page-1.png"
    assert worker._paddleocrvl("page-2.png", "ignored") == "text from page-2.png"
    assert calls == [
        ("init", {"pipeline_version": "v1"}),
        ("predict", "page-1.png"),
        ("predict", "page-2.png"),
    ]
