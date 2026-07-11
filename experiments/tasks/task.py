"""The single, spec-driven generation task.

`task_name` only names the cache dir and parallel job; the pool, oracle vs
retrieved page selection, ladder, k, and prompt modes all come from the config.
The retrieval benchmark and classifier are optional side artifacts.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from config import DEFAULT_PROMPT_MODE, ExperimentConfig
from experiments.corpus.resolve import filter_by_pool, sample_per_doc_type
from experiments.engine.side_artifacts import (
    resolve_joints,
    write_classifier_eval,
    write_retrieval_eval,
)
from experiments.tasks.base import Cell, GenerationTask, Retrievers
from pipeline.conditioner import JointTopK, OracleConditioner, RetrievedTopK
from schema import Question


def _cond_name(base: str, prompt_mode: str) -> str:
    """A conditioner name unique per prompt mode (the prediction key has no prompt
    field, so the mode must ride in the conditioner name)."""
    return f"{base}__{prompt_mode}"


def _oracle_conditioner(prompt_mode: str) -> OracleConditioner:
    cond = OracleConditioner()
    cond.name = _cond_name("oracle", prompt_mode)
    return cond


class Task(GenerationTask):
    """One spec-driven generation pass, named by its task_name label."""

    def __init__(self, name: str) -> None:
        self.name = name

    def model_specs(self, config: ExperimentConfig) -> tuple[str, ...]:
        return tuple(config.reasoner_specs) or (config.reasoner_spec,)

    def resolve_questions(self, config: ExperimentConfig, questions: Sequence[Question]) -> Sequence[Question]:
        pool = filter_by_pool(questions, config.pool)
        if config.per_doc_type_sample:
            pool = sample_per_doc_type(pool, config.per_doc_type_sample, config.sample_seed)
        return pool

    def generation_cells(self, config, questions, *, retrievers: Retrievers) -> list[Cell]:
        reps = tuple(config.representations)
        prompts = tuple(config.prompt_modes) or (DEFAULT_PROMPT_MODE,)

        if tuple(config.retrieval_representation) == ("oracle",):
            return [
                Cell(q, _oracle_conditioner(pm), rep, prompt_mode=pm)
                for q in questions for rep in reps for pm in prompts
            ]

        ks = tuple(config.k_values) or (1,)
        use_text = str(config.inference_text_retriever).lower() not in ("none", "")
        use_vision = str(config.inference_vision_retriever).lower() not in ("none", "")
        cells: list[Cell] = []
        for q in questions:
            for rep in reps:
                for pm in prompts:
                    for k in ks:
                        if use_vision:
                            cells.append(Cell(q, RetrievedTopK(
                                retrievers.vision, k, name=_cond_name(f"retrieved_vision_k{k}", pm)), rep, prompt_mode=pm))
                        if use_text:
                            cells.append(Cell(q, RetrievedTopK(
                                retrievers.text, k, name=_cond_name(f"retrieved_text_k{k}", pm)), rep, prompt_mode=pm))
                        if config.inference_joint and use_text and use_vision:
                            cells.append(Cell(q, JointTopK(
                                retrievers.text, retrievers.vision, k,
                                name=_cond_name(f"retrieved_joint_k{k}", pm)), rep, prompt_mode=pm))
        return cells

    def run_retrieval_benchmark(self, config, questions, side_dir: Path, *, limit: int | None = None) -> None:
        """Stage 1: score every configured retrieval method vs gold pages (no reasoner)."""

        pool = list(self.resolve_questions(config, questions))
        if limit is not None:
            pool = pool[:limit]
        write_retrieval_eval(
            config, pool, side_dir,
            single_ks=tuple(config.k_values) or (1,),
            joint_ks=tuple(config.joint_k_values) or (1, 3, 5),
            text_methods=config.text_retrievers,
            vision_methods=config.vision_retrievers,
            joint_pairs=resolve_joints(config.joints, config.text_retrievers, config.vision_retrievers),
            filename="retrieval.jsonl",
        )

    def run_side(self, config, questions, side_dir: Path, *, limit: int | None = None) -> None:
        """Post-inference side work: price the doc-domain classifier when configured.

        Routing prices over the answerable doc set (the docs G1 runs), independent of
        this run's own pool, and only when a classifier model is set.
        """

        if not config.classifier_spec:
            return
        docs = filter_by_pool(questions, "answerable")
        docs = sample_per_doc_type(docs, config.per_doc_type_sample, config.sample_seed)
        if limit is not None:
            docs = docs[:limit]
        write_classifier_eval(config, docs, side_dir, filename="classifier.jsonl")
