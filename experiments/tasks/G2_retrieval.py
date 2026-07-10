"""Retrieved pages across TLV/V by retrieval method and k."""

from __future__ import annotations

from pathlib import Path

from experiments.engine.side_artifacts import write_retrieval_eval
from experiments.tasks.base import Cell, GenerationTask, Retrievers, matched_cross_sweep_cells

# The reasoner-inference sweep runs at these two rungs only (pivot 7): TLV and V.
INFERENCE_REPRESENTATIONS = ("TLV", "V")
# Joint (free union) uses shallow k so the union stays under ~10 pages (pivot 4.1).
JOINT_K_VALUES = (1, 3, 5)


class G2Retrieval(GenerationTask):
    name = "G2_retrieval"
    side_artifact = "retrieval.jsonl"

    def _k_values(self, config) -> tuple[int, ...]:
        return tuple(config.k_values) if config.k_values else (1,)

    def _representations(self, config) -> tuple[str, ...]:
        reps = tuple(r for r in config.representations if r in INFERENCE_REPRESENTATIONS)
        return reps or INFERENCE_REPRESENTATIONS

    def model_specs(self, config) -> tuple[str, ...]:
        return self._reasoner_specs(config)

    def generation_cells(self, config, questions, *, retrievers: Retrievers) -> list[Cell]:
        return matched_cross_sweep_cells(
            questions,
            retrievers=retrievers,
            ks=self._k_values(config),
            joint_ks=JOINT_K_VALUES,
            representations=self._representations(config),
        )

    def run_side(self, config, questions, side_dir: Path, *, limit: int | None = None) -> None:
        """Score the full six-method + three-joint retrieval-accuracy ladder.

        This is the RQ2 accuracy benchmark (no reasoner): every text and vision
        method plus the three matched-tier joint unions, per bin, with retrieval
        cost. It is a separate experiment from the reasoner k-sweep above.

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
            joint_ks=JOINT_K_VALUES,
            filename=self.side_artifact,
        )
