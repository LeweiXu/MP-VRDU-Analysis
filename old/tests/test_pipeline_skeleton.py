"""Test the frozen Stage-3 pipeline contracts on stub implementations.

Purpose:
    Exercises the orchestrator, input conditioners, representation composers,
    model input adapters, cache keys, and modality boundary without loading real
    tools or models.

Test role:
    Ensures later stages can fill implementations behind the frozen interfaces
    without breaking cache resumability or payload invariants.

Arguments:
    None. Run with `python -m pytest tests/test_pipeline_skeleton.py`.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
import pytest

from config import ExperimentConfig, ProjectPaths
from covariates.retriever import StubRetriever
from data.loader import load_mmlongbench
from models.payload import ModelInput
from pipeline.conditioner import BuriedOracle, FullDoc, OracleConditioner, RetrievedTopK
from pipeline.orchestrator import Orchestrator, ResultCache, make_cache_key
from pipeline.reasoner import StubReasoner
from pipeline.representation import get_representation
from schema import ImagePart, Payload, Question, TextPart


@pytest.fixture(autouse=True)
def fake_representation_channels(monkeypatch) -> None:
    """Keep contract tests independent of heavy Marker/OCR layout tools."""

    import pipeline.representation as representation

    monkeypatch.setattr(
        representation,
        "text_channel",
        lambda pages: tuple(page.text or f"page {page.index}" for page in pages),
    )
    monkeypatch.setattr(
        representation,
        "layout_channel",
        lambda pages: tuple(f'{{"page": {page.index}, "blocks": []}}' for page in pages),
    )
    monkeypatch.setattr(
        representation,
        "visual_channel",
        lambda pages: tuple(ImagePart(image_path=page.image_path) for page in pages if page.image_path),
    )


def write_pdf(path: Path, pages: list[str]) -> None:
    import fitz

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def make_config(tmp_path: Path) -> ExperimentConfig:
    """Create a staged MMLongBench-like fixture and a config pointing at it."""

    data_dir = tmp_path / ".data"
    root = data_dir / "mmlongbench"
    (root / "data").mkdir(parents=True)
    (root / "documents").mkdir(parents=True)
    rows = [
        {
            "doc_id": "alpha.pdf",
            "doc_type": "Academic paper",
            "question": "Which page has the chart?",
            "answer": "page one",
            "evidence_pages": "[1]",
            "evidence_sources": "['Chart']",
            "answer_format": "String",
        },
        {
            "doc_id": "beta.pdf",
            "doc_type": "Administration/Industry file",
            "question": "What is the missing value?",
            "answer": "Not answerable",
            "evidence_pages": "[]",
            "evidence_sources": "[]",
            "answer_format": "None",
        },
    ]
    pd.DataFrame(rows).to_parquet(root / "data" / "t.parquet")
    write_pdf(root / "documents" / "alpha.pdf", ["alpha one", "alpha two", "alpha three"])
    write_pdf(root / "documents" / "beta.pdf", ["beta one", "beta two"])
    paths = ProjectPaths(root=tmp_path, data_dir=data_dir, cache_dir=tmp_path / "results" / "cache")
    return ExperimentConfig(paths=paths, dpi=72)


def all_conditioners():
    return [
        OracleConditioner(),
        FullDoc(),
        RetrievedTopK(StubRetriever(), 3),
        BuriedOracle(1),
    ]


def test_orchestrator_runs_every_cell(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    questions = load_mmlongbench(data_dir=config.paths.data_dir)
    orchestrator = Orchestrator(config, reasoner=StubReasoner())

    for question in questions:
        for conditioner in all_conditioners():
            for modality in config.representations:
                row = orchestrator.run_cell(question, conditioner, modality)
                assert row.question_id == question.id
                assert row.condition == conditioner.name
                assert row.representation == modality
                assert row.score in (0.0, 1.0)
                assert isinstance(row.correct, bool)
                # Text-only conditions never carry a visual token cost.
                if modality in ("T", "TL"):
                    assert row.input_visual_tokens == 0


def test_cache_is_idempotent_and_resumable(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    question = load_mmlongbench(data_dir=config.paths.data_dir, sample=1)[0]

    first = Orchestrator(config, reasoner=StubReasoner())
    row_a = first.run_cell(question, OracleConditioner(), "T")
    size_after_first = len(first.cache)

    # Re-running the same cell must not add a row and must return the same result.
    row_a2 = first.run_cell(question, OracleConditioner(), "T")
    assert row_a2 == row_a
    assert len(first.cache) == size_after_first

    # A fresh orchestrator resumes from the on-disk cache with no recomputation.
    second = Orchestrator(config, reasoner=StubReasoner())
    assert len(second.cache) == size_after_first
    row_b = second.run_cell(question, OracleConditioner(), "T")
    assert row_b.cache_key == row_a.cache_key


def test_cache_key_depends_on_model_spec(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    question = load_mmlongbench(data_dir=config.paths.data_dir, sample=1)[0]
    key_a = make_cache_key(question, "oracle", "T", "qwen3vl-8b-local", "stub", 72)
    key_b = make_cache_key(question, "oracle", "T", "qwen3vl-2b-local", "stub", 72)
    assert key_a != key_b


def test_modality_boundary_enforced_structurally() -> None:
    # T / TL payloads may not carry images.
    with pytest.raises(ValueError):
        Payload("T", (TextPart("x"), ImagePart(data=b"img")))
    with pytest.raises(ValueError):
        Payload("TL", (ImagePart(data=b"img"),))
    # TLV / V may.
    assert Payload("TLV", (TextPart("x"), ImagePart(data=b"img"))).image_parts
    assert Payload("V", (ImagePart(data=b"img"),)).image_parts


def test_representation_composers_respect_boundary(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    question = load_mmlongbench(data_dir=config.paths.data_dir, sample=1)[0]
    orchestrator = Orchestrator(config, reasoner=StubReasoner())
    page_set = OracleConditioner().condition(question, orchestrator.page_count(question))
    pages = orchestrator.render_pages(question, page_set)

    assert not get_representation("T").build(pages).image_parts
    assert not get_representation("TL").build(pages).image_parts
    assert get_representation("TLV").build(pages).image_parts
    assert get_representation("V").build(pages).image_parts


def test_model_input_round_trips_through_both_adapters() -> None:
    original = ModelInput(
        (TextPart("describe the figure"), ImagePart(data=b"\x89PNG-bytes"), TextPart("tail"))
    )

    # Chat adapter: lossless round-trip of text and image content.
    restored = ModelInput.from_chat_messages(original.to_chat_messages())
    assert [p.text for p in restored.text_parts] == ["describe the figure", "tail"]
    assert restored.image_parts[0].read_bytes() == b"\x89PNG-bytes"

    # Local adapter: one placeholder per image, images returned in order.
    prompt, images = original.to_local_prompt()
    assert prompt.count("<image>") == 1
    assert len(images) == 1
    assert images[0].read_bytes() == b"\x89PNG-bytes"


def test_image_part_data_uri_is_base64() -> None:
    part = ImagePart(data=b"hello", mime="image/png")
    uri = part.data_uri()
    assert uri.startswith("data:image/png;base64,")
    encoded = uri.split(",", 1)[1]
    assert base64.b64decode(encoded) == b"hello"
