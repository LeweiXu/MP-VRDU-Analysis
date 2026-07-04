"""Expand experiment configs into cached pipeline cells and policy rows.

Purpose:
    Provides reusable cell-expansion helpers for MVP smoke stages before the
    full paper-table runner exists. Stage M4 uses the oracle ladder helper;
    Stage M6 adds covariate smoke helpers for matched-vs-cross retrieval and
    routing policies.

Pipeline role:
    Keeps experiment expansion outside `pipeline.orchestrator`: this module
    chooses questions, conditioners, representation rungs, retrievers, and
    routing policies, while the orchestrator remains the single-cell
    A->B->C->D executor and cache owner. Section-2 stages scale these helpers to
    full grids without changing the frozen single-cell run loop.

Arguments:
    None. This module is import-only; callers pass an `ExperimentConfig`, an
    iterable of `Question` objects, and optionally injected orchestrators,
    retrievers, classifiers, or recipe maps.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from config import ExperimentConfig
from covariates.classifier import DocTypeClassifier, DocTypePrediction, QwenDocTypeClassifier
from covariates.retriever import BM25BGERetriever, ColQwenRetriever, MemoizedRetriever, Retriever
from data.binning import DocTypeBin, doc_type_bin
from metrics.accuracy import accuracy_summary
from metrics.cost import cost_summary
from metrics.retrieval import RetrievalEvalRow, score_retrieval
from pipeline.conditioner import OracleConditioner, RetrievedTopK
from pipeline.orchestrator import Orchestrator, ResultRow, make_cache_key
from schema import Modality, Question


DEFAULT_ROUTING_RECIPES: Mapping[DocTypeBin, Modality] = {
    "text_heavy": "T",
    "in_between": "TL",
    "visual_heavy": "TLV",
}


@dataclass(frozen=True)
class RunBatch:
    """Rows plus cache accounting from one expanded orchestrator pass."""

    rows: tuple[ResultRow, ...]
    cache_hits: int
    computed: int
    cache_path: Path
    cache_rows: int


@dataclass(frozen=True)
class MatchedCrossBatch:
    """Rows and retrieval metrics from the Stage-M6 matched/cross smoke."""

    rows: tuple[ResultRow, ...]
    retrieval_rows: tuple[RetrievalEvalRow, ...]
    cache_hits: int
    computed: int
    cache_path: Path
    cache_rows: int


@dataclass(frozen=True)
class ClassificationLogRow:
    """One doc-level classifier prediction used by predicted routing."""

    question_id: str
    doc_id: str
    gold_doc_type: str
    predicted_doc_type: str
    gold_bin: str
    predicted_bin: str
    correct_bin: bool
    confidence: float
    latency_s: float
    raw_text: str
    classifier: str


@dataclass(frozen=True)
class RoutingPolicyRow:
    """Corpus-level policy summary with classifier latency accounted separately."""

    policy: str
    n_rows: int
    n_docs: int
    accuracy: float
    ci_low: float
    ci_high: float
    latency_bs1_s: float
    classifier_latency_bs1_s: float
    total_latency_bs1_s: float
    text_tokens: int
    vision_tokens: int
    output_tokens: int
    chosen_rungs: tuple[str, ...]


@dataclass(frozen=True)
class RoutingPolicyBatch:
    """Policy summaries plus cached result rows and classifier logs."""

    policy_rows: tuple[RoutingPolicyRow, ...]
    result_rows: tuple[ResultRow, ...]
    classifier_rows: tuple[ClassificationLogRow, ...]
    cache_path: Path
    cache_rows: int


def run_oracle_ladder(
    config: ExperimentConfig,
    questions: Iterable[Question],
    *,
    orchestrator: Orchestrator | None = None,
    representations: Sequence[Modality] | None = None,
) -> RunBatch:
    """Run oracle pages through every requested representation rung.

    Cache hits are counted before calling `run_cell()`. A second call with the
    same orchestrator/cache should therefore report `computed == 0`.
    """

    orchestrator = orchestrator or Orchestrator(config)
    conditioner = OracleConditioner()
    rungs = tuple(representations or config.representations)
    rows: list[ResultRow] = []
    cache_hits = 0

    for question in questions:
        for representation in rungs:
            key = make_cache_key(
                question,
                conditioner.name,
                representation,
                orchestrator.reasoner.spec,
                orchestrator.judge.spec,
                config.dpi,
            )
            if orchestrator.cache.get(key) is not None:
                cache_hits += 1
            rows.append(orchestrator.run_cell(question, conditioner, representation))

    return RunBatch(
        rows=tuple(rows),
        cache_hits=cache_hits,
        computed=len(rows) - cache_hits,
        cache_path=orchestrator.cache.path,
        cache_rows=len(orchestrator.cache),
    )


def run_matched_cross_smoke(
    config: ExperimentConfig,
    questions: Iterable[Question],
    *,
    orchestrator: Orchestrator | None = None,
    text_retriever: Retriever | None = None,
    vision_retriever: Retriever | None = None,
    k: int | None = None,
    representation: Modality = "TLV",
) -> MatchedCrossBatch:
    """Run M6 matched/cross retrieval pipelines through the orchestrator.

    Matched uses vision retrieval plus a vision-bearing representation. Cross
    uses text retrieval plus the same vision-bearing representation. Retrieval
    metrics are recorded before the normal `RetrievedTopK` conditioner feeds the
    selected pages to the reasoner.
    """

    orchestrator = orchestrator or Orchestrator(config)
    top_k = int(k if k is not None else (config.k_values[0] if config.k_values else 1))
    text = MemoizedRetriever(
        text_retriever
        or BM25BGERetriever(
            data_dir=config.paths.data_dir,
            cache_dir=config.paths.cache_dir,
            dpi=config.dpi,
        )
    )
    vision = MemoizedRetriever(
        vision_retriever
        or ColQwenRetriever(
            data_dir=config.paths.data_dir,
            cache_dir=config.paths.cache_dir,
            dpi=config.dpi,
        )
    )
    pipelines = (
        ("matched_vision", "vision", vision, RetrievedTopK(vision, top_k, name=f"retrieved_vision_k{top_k}")),
        ("cross_text_to_vision", "text", text, RetrievedTopK(text, top_k, name=f"retrieved_text_k{top_k}")),
    )

    rows: list[ResultRow] = []
    retrieval_rows: list[RetrievalEvalRow] = []
    cache_hits = 0
    for question in questions:
        page_count = orchestrator.page_count(question)
        for _pipeline, modality, retriever, conditioner in pipelines:
            retrieved = retriever.retrieve(question, page_count, top_k)
            retrieval_rows.append(
                score_retrieval(
                    question,
                    retrieved,
                    retriever=retriever.name,
                    modality=modality,
                    k=top_k,
                )
            )
            key = make_cache_key(
                question,
                conditioner.name,
                representation,
                orchestrator.reasoner.spec,
                orchestrator.judge.spec,
                config.dpi,
            )
            if orchestrator.cache.get(key) is not None:
                cache_hits += 1
            row = orchestrator.run_cell(question, conditioner, representation)
            rows.append(row)

    return MatchedCrossBatch(
        rows=tuple(rows),
        retrieval_rows=tuple(retrieval_rows),
        cache_hits=cache_hits,
        computed=len(rows) - cache_hits,
        cache_path=orchestrator.cache.path,
        cache_rows=len(orchestrator.cache),
    )


def _prediction_log(question: Question, prediction: DocTypePrediction, classifier: DocTypeClassifier) -> ClassificationLogRow:
    """Return a classifier log row with gold/predicted bins."""

    gold_bin = doc_type_bin(question.doc_type)
    predicted_bin = str(prediction.bin or gold_bin)
    return ClassificationLogRow(
        question_id=question.id,
        doc_id=question.doc_id,
        gold_doc_type=question.doc_type,
        predicted_doc_type=prediction.doc_type,
        gold_bin=gold_bin,
        predicted_bin=predicted_bin,
        correct_bin=predicted_bin == gold_bin,
        confidence=prediction.confidence,
        latency_s=prediction.latency_s,
        raw_text=prediction.raw_text,
        classifier=classifier.name,
    )


def _policy_summary(
    policy: str,
    rows: Sequence[ResultRow],
    *,
    classifier_latency_total_s: float = 0.0,
) -> RoutingPolicyRow:
    """Aggregate one routing policy into a corpus-level row."""

    acc = accuracy_summary(rows, n_bootstrap=0)
    cost = cost_summary(rows)
    n_rows = len(rows)
    classifier_latency_bs1_s = classifier_latency_total_s / n_rows if n_rows else 0.0
    return RoutingPolicyRow(
        policy=policy,
        n_rows=acc.n_rows,
        n_docs=acc.n_docs,
        accuracy=acc.accuracy,
        ci_low=acc.ci_low,
        ci_high=acc.ci_high,
        latency_bs1_s=cost.latency_bs1_s,
        classifier_latency_bs1_s=classifier_latency_bs1_s,
        total_latency_bs1_s=cost.latency_bs1_s + classifier_latency_bs1_s,
        text_tokens=cost.input_text_tokens,
        vision_tokens=cost.input_visual_tokens,
        output_tokens=cost.output_tokens,
        chosen_rungs=tuple(row.representation for row in rows),
    )


def run_routing_policies_smoke(
    config: ExperimentConfig,
    questions: Iterable[Question],
    *,
    orchestrator: Orchestrator | None = None,
    classifier: DocTypeClassifier | None = None,
    recipe_by_bin: Mapping[str, Modality] | None = None,
) -> RoutingPolicyBatch:
    """Run the four M6 routing policies and return corpus-level summaries."""

    question_list = tuple(questions)
    orchestrator = orchestrator or Orchestrator(config)
    classifier = classifier or QwenDocTypeClassifier(
        data_dir=config.paths.data_dir,
        cache_dir=config.paths.cache_dir,
        dpi=config.dpi,
    )
    recipe = dict(DEFAULT_ROUTING_RECIPES)
    if recipe_by_bin is not None:
        recipe.update({str(key): value for key, value in recipe_by_bin.items()})

    classifier_predictions: dict[str, DocTypePrediction] = {}
    classifier_logs: list[ClassificationLogRow] = []
    for question in question_list:
        if question.doc_id in classifier_predictions:
            continue
        prediction = classifier.classify(question)
        classifier_predictions[question.doc_id] = prediction
        classifier_logs.append(_prediction_log(question, prediction, classifier))

    oracle = OracleConditioner()
    all_rows: list[ResultRow] = []
    policy_rows: list[RoutingPolicyRow] = []
    policies = ("oracle_routing", "predicted_routing", "uniform_cheapest_T", "uniform_strongest_TLV")
    for policy in policies:
        rows: list[ResultRow] = []
        for question in question_list:
            if policy == "oracle_routing":
                rung = recipe[doc_type_bin(question.doc_type)]
            elif policy == "predicted_routing":
                predicted_bin = str(classifier_predictions[question.doc_id].bin)
                rung = recipe.get(predicted_bin, recipe[doc_type_bin(question.doc_type)])
            elif policy == "uniform_cheapest_T":
                rung = "T"
            elif policy == "uniform_strongest_TLV":
                rung = "TLV"
            else:
                raise ValueError(f"unknown routing policy {policy!r}")
            rows.append(orchestrator.run_cell(question, oracle, rung))
        all_rows.extend(rows)
        classifier_latency = (
            sum(prediction.latency_s for prediction in classifier_predictions.values())
            if policy == "predicted_routing"
            else 0.0
        )
        policy_rows.append(_policy_summary(policy, rows, classifier_latency_total_s=classifier_latency))

    return RoutingPolicyBatch(
        policy_rows=tuple(policy_rows),
        result_rows=tuple(all_rows),
        classifier_rows=tuple(classifier_logs),
        cache_path=orchestrator.cache.path,
        cache_rows=len(orchestrator.cache),
    )
