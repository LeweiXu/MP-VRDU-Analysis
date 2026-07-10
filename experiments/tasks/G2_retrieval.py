"""Retrieved pages across TLV/V by retrieval method and k."""

from __future__ import annotations

from pathlib import Path

from experiments.engine.side_artifacts import resolve_joints, write_retrieval_eval
from experiments.tasks.base import Cell, GenerationTask, Retrievers, matched_cross_sweep_cells


class G2Retrieval(GenerationTask):
    name = "G2_retrieval"
    side_artifact = "retrieval.jsonl"

    def _k_values(self, config) -> tuple[int, ...]:
        return tuple(config.k_values) if config.k_values else (1,)

    def _representations(self, config) -> tuple[str, ...]:
        return tuple(config.inference_representations) or ("TLV", "V")

    def model_specs(self, config) -> tuple[str, ...]:
        return self._reasoner_specs(config)

    def generation_cells(self, config, questions, *, retrievers: Retrievers) -> list[Cell]:
        return matched_cross_sweep_cells(
            questions,
            retrievers=retrievers,
            ks=self._k_values(config),
            joint_ks=tuple(config.joint_k_values),
            representations=self._representations(config),
            include_joint=config.inference_joint,
        )

    def run_side(self, config, questions, side_dir: Path, *, limit: int | None = None) -> None:
        """Score the retrieval-accuracy ladder (page P/R/F1 + cost), no reasoner.

        This is the RQ2 accuracy benchmark: every configured text and vision method
        plus the joint unions, per bin, with retrieval cost. The method sets come
        from the spec's G2 `retrieval` block (`config.text_retrievers` /
        `vision_retrievers` / `joints`). It is a separate experiment from the
        reasoner k-sweep above.

        `questions` is the full corpus, so re-resolve G2's own scope (answerable
        pool + per_doc_type sample) before scoring, and apply the smoke `limit`.
        """

        pool = list(self.resolve_questions(config, questions))
        if limit is not None:
            pool = pool[:limit]
        write_retrieval_eval(
            config,
            pool,
            side_dir,
            single_ks=self._k_values(config),
            joint_ks=tuple(config.joint_k_values),
            text_methods=config.text_retrievers,
            vision_methods=config.vision_retrievers,
            joint_pairs=resolve_joints(config.joints, config.text_retrievers, config.vision_retrievers),
            filename=self.side_artifact,
        )
