"""Test the Stage-M4 oracle ladder end to end through the cache.

Purpose:
    Proves the MVP A->B->C path for oracle pages: selected evidence pages are
    rendered, all four representation rungs produce backend-neutral
    `ModelInput`, the reasoner receives the expected text/image boundary, and
    the orchestrator writes one resumable cached row per smoke question/rung.

Test role:
    Uses lightweight fixture PDFs plus injected Marker-like text/layout channels
    and a recording reasoner, so the test covers orchestrator wiring and cache
    behavior without loading Marker or Qwen weights.

Arguments:
    None. Run with `python -m pytest tests/test_ladder_e2e.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import ExperimentConfig, ProjectPaths
from data.loader import load_mmlongbench
from experiments.base import oracle_ladder_cells
from models.payload import ModelInput
from pipeline.orchestrator import Orchestrator, ResultCache
from pipeline.reasoner import Reasoner
from schema import ImagePart, Prediction, Question, TextPart


@dataclass
class _LadderBatch:
    """Rows plus cache accounting, mirroring the old runner helper."""

    rows: tuple
    computed: int
    cache_hits: int
    cache_rows: int
    cache_path: object


def run_oracle_ladder(config, questions, *, orchestrator, representations=None):
    """Run the oracle ladder through the orchestrator (test helper).

    Replaces the removed `experiments.runner.run_oracle_ladder`; the production
    path is now `experiments.driver` over the `G1_sufficiency` task.
    """

    reps = tuple(representations) if representations is not None else config.representations
    rows = []
    cache_hits = 0
    for cell in oracle_ladder_cells(config, questions):
        if cell.representation not in reps:
            continue
        from pipeline.orchestrator import make_cache_key

        key = make_cache_key(
            cell.question,
            cell.conditioner.name,
            cell.representation,
            orchestrator.reasoner.spec,
            orchestrator.judge.spec,
            config.dpi,
        )
        if orchestrator.cache.get(key) is not None:
            cache_hits += 1
        rows.append(orchestrator.run_cell(cell.question, cell.conditioner, cell.representation))
    return _LadderBatch(
        rows=tuple(rows),
        computed=len(rows) - cache_hits,
        cache_hits=cache_hits,
        cache_rows=len(orchestrator.cache),
        cache_path=orchestrator.cache.path,
    )


class RecordingReasoner(Reasoner):
    """Reasoner fake that records every model input it is asked to answer."""

    def __init__(self, spec: str = "m4-recording") -> None:
        self.spec = spec
        self.inputs: list[tuple[str, ModelInput]] = []

    def answer(self, question: Question, model_input: ModelInput) -> Prediction:
        self.inputs.append((question.id, model_input))
        text_tokens = sum(len(part.text.split()) for part in model_input.text_parts)
        visual_tokens = 17 * len(model_input.image_parts)
        return Prediction(
            text=f"{question.gold_answer} synthetic",
            model_spec=self.spec,
            input_text_tokens=text_tokens,
            input_visual_tokens=visual_tokens,
            output_tokens=2,
            latency_s=0.001,
            metadata={
                "n_text_parts": len(model_input.text_parts),
                "n_image_parts": len(model_input.image_parts),
            },
        )


def write_pdf(path: Path, pages: list[str]) -> None:
    """Write a tiny PDF fixture."""

    import fitz

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def make_config(tmp_path: Path) -> ExperimentConfig:
    """Create a small staged MMLongBench-like fixture."""

    data_dir = tmp_path / ".data"
    root = data_dir / "mmlongbench"
    (root / "data").mkdir(parents=True)
    (root / "documents").mkdir(parents=True)
    rows = [
        {
            "doc_id": "smoke-alpha.pdf",
            "doc_type": "Academic paper",
            "question": "Which value is on the evidence page?",
            "answer": "alpha answer",
            "evidence_pages": "[1]",
            "evidence_sources": "['Text', 'Chart']",
            "answer_format": "String",
        },
        {
            "doc_id": "smoke-beta.pdf",
            "doc_type": "Financial report",
            "question": "What number appears in the table?",
            "answer": "beta answer",
            "evidence_pages": "[2]",
            "evidence_sources": "['Table']",
            "answer_format": "String",
        },
    ]
    pd.DataFrame(rows).to_parquet(root / "data" / "smoke.parquet")
    write_pdf(root / "documents" / "smoke-alpha.pdf", ["alpha answer", "alpha extra"])
    write_pdf(root / "documents" / "smoke-beta.pdf", ["beta cover", "beta answer"])
    paths = ProjectPaths(root=tmp_path, data_dir=data_dir, cache_dir=tmp_path / "results" / "cache")
    return ExperimentConfig(smoke=True, paths=paths, dpi=72)


def test_oracle_ladder_rows_payload_boundaries_and_cache_hits(tmp_path: Path, monkeypatch) -> None:
    import pipeline.representation as representation

    def fake_text_channel(pages):
        return tuple(f"marker text page={page.index}: {page.text}" for page in pages)

    def fake_layout_channel(pages):
        return tuple(
            json.dumps(
                {
                    "source": "marker",
                    "page_index": page.index,
                    "blocks": [{"type": "Text", "text": page.text, "bbox": [0, 0, 10, 10]}],
                },
                sort_keys=True,
            )
            for page in pages
        )

    monkeypatch.setattr(representation, "text_channel", fake_text_channel)
    monkeypatch.setattr(representation, "layout_channel", fake_layout_channel)

    config = make_config(tmp_path)
    questions = load_mmlongbench(config.paths.data_dir)
    reasoner = RecordingReasoner()
    orchestrator = Orchestrator(config, reasoner=reasoner)

    batch = run_oracle_ladder(config, questions, orchestrator=orchestrator)

    assert len(batch.rows) == len(questions) * len(config.representations)
    assert batch.computed == len(batch.rows)
    assert batch.cache_hits == 0
    assert batch.cache_rows == len(batch.rows)
    assert len(orchestrator.cache) == len(batch.rows)
    assert batch.cache_path == config.paths.cache_dir / "orchestrator" / "results.jsonl"

    seen_inputs: dict[tuple[str, str], ModelInput] = {}
    for row, (_, model_input) in zip(batch.rows, reasoner.inputs, strict=True):
        assert row.condition == "oracle"
        assert row.provenance == "oracle"
        assert row.model_spec == reasoner.spec
        assert row.cache_key
        assert row.page_indices
        seen_inputs[(row.question_id, row.representation)] = model_input
        if row.representation in ("T", "TL"):
            assert row.input_visual_tokens == 0
            assert not model_input.image_parts
        else:
            assert row.input_visual_tokens > 0
            assert model_input.image_parts
            assert all(part.image_path and part.image_path.exists() for part in model_input.image_parts)

    for question in questions:
        t_input = seen_inputs[(question.id, "T")]
        tl_input = seen_inputs[(question.id, "TL")]
        tlv_input = seen_inputs[(question.id, "TLV")]
        v_input = seen_inputs[(question.id, "V")]

        assert [type(part) for part in t_input.parts] == [TextPart]
        assert "marker text" in t_input.text_parts[0].text
        assert [type(part) for part in tl_input.parts] == [TextPart, TextPart]
        assert "marker text" in tl_input.text_parts[0].text
        assert '"source": "marker"' in tl_input.text_parts[1].text
        assert tlv_input.text_parts and tlv_input.image_parts
        assert all(isinstance(part, ImagePart) for part in v_input.parts)

    first_call_count = len(reasoner.inputs)
    repeat = run_oracle_ladder(config, questions, orchestrator=orchestrator)

    assert repeat.rows == batch.rows
    assert repeat.computed == 0
    assert repeat.cache_hits == len(batch.rows)
    assert len(reasoner.inputs) == first_call_count

    fresh_reasoner = RecordingReasoner()
    fresh_orchestrator = Orchestrator(config, reasoner=fresh_reasoner)
    resumed = run_oracle_ladder(config, questions, orchestrator=fresh_orchestrator)

    assert resumed.rows == batch.rows
    assert resumed.computed == 0
    assert resumed.cache_hits == len(batch.rows)
    assert fresh_reasoner.inputs == []


def test_oracle_ladder_cache_key_includes_model_spec(tmp_path: Path, monkeypatch) -> None:
    import pipeline.representation as representation

    monkeypatch.setattr(representation, "text_channel", lambda pages: tuple(page.text for page in pages))
    monkeypatch.setattr(representation, "layout_channel", lambda pages: tuple("{}" for _ in pages))

    config = make_config(tmp_path)
    question = load_mmlongbench(config.paths.data_dir, sample=1)
    shared_cache = ResultCache(config.paths.cache_dir / "orchestrator" / "results.jsonl")

    first_reasoner = RecordingReasoner("m4-model-a")
    first = Orchestrator(config, reasoner=first_reasoner, cache=shared_cache)
    row_a = run_oracle_ladder(config, question, orchestrator=first, representations=("T",)).rows[0]

    second_reasoner = RecordingReasoner("m4-model-b")
    second = Orchestrator(
        config,
        reasoner=second_reasoner,
        cache=ResultCache(config.paths.cache_dir / "orchestrator" / "results.jsonl"),
    )
    row_b = run_oracle_ladder(config, question, orchestrator=second, representations=("T",)).rows[0]

    assert row_a.cache_key != row_b.cache_key
    assert row_a.model_spec == "m4-model-a"
    assert row_b.model_spec == "m4-model-b"
    assert len(second.cache) == 2
