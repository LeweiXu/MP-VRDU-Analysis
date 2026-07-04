"""Test Stage-M6 retrieval, classifier, matched/cross, and routing covariates.

Purpose:
    Verifies the final MVP covariate layer: text and vision retrievers return
    ranked smoke pages, retrieval metrics score page overlap, the doc-type
    classifier returns a valid Option-A bin through an injected reasoner, and
    matched/cross plus routing-policy runners produce corpus-level artifacts.

Test role:
    Uses tiny fixture PDFs plus fake reasoners/retrievers so M6 contracts are
    covered locally without downloading BGE, ColQwen, Marker, or Qwen weights.

Arguments:
    None. Run with `python -m pytest tests/test_covariates.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from config import ExperimentConfig, ProjectPaths
from covariates.classifier import DocTypeClassifier, DocTypePrediction, QwenDocTypeClassifier
from covariates.retriever import BM25BGERetriever, ColQwenRetriever, Retriever
from data.binning import doc_type_bin
from data.loader import load_mmlongbench
from experiments.runner import run_matched_cross_smoke, run_routing_policies_smoke
from metrics.retrieval import (
    page_prf,
    retrieval_slice_keys,
    retrieval_summary_by_evidence_slice,
    retrieval_summary_by_modality,
    score_retrieval,
)
from models.payload import ModelInput
from pipeline.orchestrator import Orchestrator
from pipeline.reasoner import Reasoner
from schema import Prediction, Question


@pytest.fixture(autouse=True)
def fake_representation_channels(monkeypatch) -> None:
    """Keep M6 runner tests independent of Marker while preserving boundaries."""

    import pipeline.representation as representation

    monkeypatch.setattr(
        representation,
        "text_channel",
        lambda pages: tuple(page.text or f"page {page.index}" for page in pages),
    )
    monkeypatch.setattr(
        representation,
        "layout_channel",
        lambda pages: tuple(
            json.dumps({"source": "test", "page_index": page.index, "blocks": []})
            for page in pages
        ),
    )


class GoldAnswerReasoner(Reasoner):
    """Reasoner fake that answers with the schema gold answer and cost fields."""

    def __init__(self, spec: str = "m6-gold") -> None:
        self.spec = spec
        self.inputs: list[tuple[str, ModelInput]] = []

    def answer(self, question: Question, model_input: ModelInput) -> Prediction:
        self.inputs.append((question.id, model_input))
        return Prediction(
            text=f"{question.gold_answer} from fixture",
            model_spec=self.spec,
            input_text_tokens=sum(len(part.text.split()) for part in model_input.text_parts),
            input_visual_tokens=11 * len(model_input.image_parts),
            output_tokens=3,
            latency_s=0.2 + 0.1 * len(model_input.image_parts),
        )


class StaticReasoner(Reasoner):
    """Reasoner fake that always emits the same classifier label."""

    def __init__(self, text: str) -> None:
        self.spec = "m6-static"
        self.text = text

    def answer(self, question: Question, model_input: ModelInput) -> Prediction:
        return Prediction(
            text=self.text,
            model_spec=self.spec,
            input_text_tokens=7,
            input_visual_tokens=13 * len(model_input.image_parts),
            output_tokens=1,
            latency_s=0.4,
        )


class GoldRetriever(Retriever):
    """Retriever fake that always ranks the gold evidence page first."""

    def __init__(self, name: str) -> None:
        self.name = name

    def retrieve(self, question: Question, page_count: int, k: int) -> tuple[int, ...]:
        ranked = list(question.evidence_pages)
        ranked.extend(page for page in range(page_count) if page not in ranked)
        return tuple(ranked[:k])


class LatencyClassifier(DocTypeClassifier):
    """Classifier fake that predicts the gold bin with fixed doc-level latency."""

    name = "latency_classifier"

    def __init__(self, latency_s: float = 0.3) -> None:
        self.latency_s = latency_s

    def classify(self, question: Question) -> DocTypePrediction:
        bin_name = doc_type_bin(question.doc_type)
        return DocTypePrediction(
            doc_type=question.doc_type,
            confidence=1.0,
            bin=bin_name,
            latency_s=self.latency_s,
            raw_text=question.doc_type,
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
    """Create a three-bin MMLongBench-like fixture corpus."""

    data_dir = tmp_path / ".data"
    root = data_dir / "mmlongbench"
    (root / "data").mkdir(parents=True)
    (root / "documents").mkdir(parents=True)
    rows = [
        {
            "doc_id": "text.pdf",
            "doc_type": "Academic paper",
            "question": "Where is the alpha needle?",
            "answer": "alpha needle answer",
            "evidence_pages": "[2]",
            "evidence_sources": "['Text']",
            "answer_format": "String",
        },
        {
            "doc_id": "mid.pdf",
            "doc_type": "Financial report",
            "question": "Which beta table value is reported?",
            "answer": "beta table answer",
            "evidence_pages": "[1]",
            "evidence_sources": "['Table']",
            "answer_format": "String",
        },
        {
            "doc_id": "visual.pdf",
            "doc_type": "Brochure",
            "question": "Which gamma visual caption appears?",
            "answer": "gamma visual answer",
            "evidence_pages": "[2]",
            "evidence_sources": "['Figure', 'Chart']",
            "answer_format": "String",
        },
    ]
    pd.DataFrame(rows).to_parquet(root / "data" / "fixture.parquet")
    write_pdf(root / "documents" / "text.pdf", ["cover", "alpha needle answer is here", "tail"])
    write_pdf(root / "documents" / "mid.pdf", ["beta table answer is here", "tail"])
    write_pdf(root / "documents" / "visual.pdf", ["cover", "gamma visual caption answer", "tail"])
    paths = ProjectPaths(root=tmp_path, data_dir=data_dir, cache_dir=tmp_path / "results" / "cache")
    return ExperimentConfig(smoke=True, paths=paths, dpi=72)


def test_retrievers_return_ranked_pages_and_metrics_slice_by_modality(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    questions = load_mmlongbench(config.paths.data_dir)
    question = questions[0]

    text_retriever = BM25BGERetriever(
        data_dir=config.paths.data_dir,
        cache_dir=config.paths.cache_dir,
        dpi=config.dpi,
        use_bge=False,
    )
    assert text_retriever.retrieve(question, page_count=3, k=2)[0] == 1

    vision_retriever = ColQwenRetriever(
        data_dir=config.paths.data_dir,
        cache_dir=config.paths.cache_dir,
        dpi=config.dpi,
        scorer=lambda _question, pages: [1.0 if page.index == 1 else 0.0 for page in pages],
    )
    assert vision_retriever.retrieve(question, page_count=3, k=2)[0] == 1

    perfect = page_prf((1,), (1,))
    assert perfect.precision == 1.0
    assert perfect.recall == 1.0
    assert perfect.f1 == 1.0

    row = score_retrieval(question, (1,), retriever="toy", modality="text", k=1)
    assert retrieval_summary_by_modality([row])["text"].f1 == 1.0
    assert retrieval_slice_keys(question, "text") == ("text:text",)
    assert retrieval_summary_by_evidence_slice([row])["text:text"].recall == 1.0


def test_qwen_doc_type_classifier_returns_valid_bin(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    question = load_mmlongbench(config.paths.data_dir)[2]
    classifier = QwenDocTypeClassifier(
        data_dir=config.paths.data_dir,
        cache_dir=config.paths.cache_dir,
        dpi=config.dpi,
        reasoner=StaticReasoner("Brochure"),
    )

    prediction = classifier.classify(question)

    assert prediction.doc_type == "Brochure"
    assert prediction.bin == "visual_heavy"
    assert prediction.latency_s == 0.4
    assert prediction.metadata["correct_bin"] is True
    assert prediction.metadata["prompt_version"] == "m6-doc-type-classifier-v1"


def test_matched_cross_smoke_runs_retrieval_and_reasoning(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    questions = load_mmlongbench(config.paths.data_dir)
    orchestrator = Orchestrator(config, reasoner=GoldAnswerReasoner())

    batch = run_matched_cross_smoke(
        config,
        questions,
        orchestrator=orchestrator,
        text_retriever=GoldRetriever("fake_text"),
        vision_retriever=GoldRetriever("fake_vision"),
        k=1,
    )

    assert len(batch.rows) == len(questions) * 2
    assert len(batch.retrieval_rows) == len(questions) * 2
    assert {row.modality for row in batch.retrieval_rows} == {"text", "vision"}
    assert all(row.f1 == 1.0 for row in batch.retrieval_rows)
    assert {row.condition for row in batch.rows} == {"retrieved_text_k1", "retrieved_vision_k1"}
    assert all(row.representation == "TLV" for row in batch.rows)


def test_routing_policies_produce_corpus_level_rows_with_classifier_cost(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    questions = load_mmlongbench(config.paths.data_dir)
    orchestrator = Orchestrator(config, reasoner=GoldAnswerReasoner())

    batch = run_routing_policies_smoke(
        config,
        questions,
        orchestrator=orchestrator,
        classifier=LatencyClassifier(latency_s=0.3),
        recipe_by_bin={
            "text_heavy": "T",
            "in_between": "TL",
            "visual_heavy": "TLV",
        },
    )

    policies = {row.policy: row for row in batch.policy_rows}
    assert set(policies) == {
        "oracle_routing",
        "predicted_routing",
        "uniform_cheapest_T",
        "uniform_strongest_TLV",
    }
    assert all(row.n_rows == len(questions) for row in policies.values())
    assert all(row.accuracy == 1.0 for row in policies.values())
    assert len(batch.classifier_rows) == len(questions)
    assert policies["predicted_routing"].classifier_latency_bs1_s == pytest.approx(0.3)
    assert policies["predicted_routing"].total_latency_bs1_s == pytest.approx(
        policies["predicted_routing"].latency_bs1_s + 0.3
    )
    assert policies["uniform_cheapest_T"].chosen_rungs == ("T", "T", "T")
    assert policies["uniform_strongest_TLV"].chosen_rungs == ("TLV", "TLV", "TLV")
