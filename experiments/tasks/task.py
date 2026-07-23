"""The single, spec-driven generation task.

`task_name` only names the cache dir and parallel job; the pool, oracle vs
retrieved page selection, ladder, k, and prompt modes all come from the config.
The retrieval benchmark and classifier are optional side artifacts.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from config import DEFAULT_PROMPT_MODE, ExperimentConfig
from experiments.corpus.resolve import (
    filter_by_hop,
    filter_by_pool,
    filter_by_scan,
    resolve_corpus,
    sample_per_doc_type,
)
from experiments.engine.side_artifacts import (
    resolve_joints,
    write_classifier_eval,
    write_retrieval_eval,
)
from experiments.tasks.base import Cell, GenerationTask, Retrievers
from pipeline.conditioner import JointTopK, OracleConditioner, PageSetConditioner, RetrievedTopK
from schema import Question

log = logging.getLogger("mpvrdu.experiments")


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
        # Scan filter (digital/scanned) comes first, before the pool and sampling.
        qs: Sequence[Question] = questions
        if getattr(config, "scan_filter", "any") not in ("any", None, ""):
            qs = filter_by_scan(qs, config.scan_filter, data_dir=config.paths.data_dir,
                                annotations_dir=config.paths.root / "annotations")
        pool = filter_by_pool(qs, config.pool)
        # Hop filter (single/multi gold-evidence pages) after the pool, so a
        # page_set run's gold rules always see the gold count they need.
        pool = filter_by_hop(pool, getattr(config, "hop_filter", "any"))
        # Sampling (full / per_doc_type / per_bin / limit / ids) runs last, over the
        # scan-, pool-, and hop-filtered subset.
        return resolve_corpus({"sampling": config.sampling}, pool)

    def generation_cells(self, config, questions, *, retrievers: Retrievers) -> list[Cell]:
        reps = tuple(config.representations)
        prompts = tuple(config.prompt_modes) or (DEFAULT_PROMPT_MODE,)

        if getattr(config, "page_set", None):
            return self._page_set_cells(config, questions, retrievers, reps, prompts)

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

    def _page_set_cells(self, config, questions, retrievers: Retrievers, reps, prompts) -> list[Cell]:
        """Cells for a declared page_set: rules = ranking_sources x distractor counts.

        Count-decidable degenerate (rule, question) pairs are excluded here by
        documented policy and logged; ranking-dependent problems surface at
        condition time as error status rows (`PageSetRuleError`).
        """

        from collections import Counter

        from pipeline.page_rules import PageSetRule, encode_base, enumeration_skip_reason

        block = config.page_set
        rules = [
            PageSetRule(
                ranking_source=str(ranker),
                gold_mode=str(block["gold"]["mode"]),
                gold_count=int(block["gold"]["count"]),
                distractor_count=int(d),
                on_insufficient_gold=str(block["on_insufficient_gold"]),
                on_insufficient_distractors=str(block["on_insufficient_distractors"]),
                on_no_gold=str(block["on_no_gold"]),
            )
            for ranker in block["ranking_source"]
            for d in block["distractor"]["count"]
        ]
        skipped: Counter[str] = Counter()
        cells: list[Cell] = []
        for q in questions:
            for rule in rules:
                reason = enumeration_skip_reason(rule, q)
                if reason:
                    skipped[reason] += 1
                    continue
                ranker = retrievers.rankers[rule.ranking_source]
                base = encode_base(rule)
                for rep in reps:
                    for pm in prompts:
                        cells.append(Cell(
                            q, PageSetConditioner(ranker, rule, name=_cond_name(base, pm)),
                            rep, prompt_mode=pm,
                        ))
        if skipped:
            log.info("page_set %s: excluded by policy: %s", self.name, dict(skipped))
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
