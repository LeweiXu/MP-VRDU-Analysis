"""Main experiment command-line entry point for cached pipeline runs.

Stage 3 wiring: expand the config's conditions x representations over a tiny
sample of questions, run each cell through the orchestrator (on stubs), and print
a summary. Real tools and models fill in behind the frozen interfaces in later
stages without changing this entry point. The heavy grid expansion for the paper
tables is the Section-2 `experiments/runner.py`; this CLI is the minimal
end-to-end smoke (the v3 MVP adds a `--smoke` corpus selector on top).

    python -m cli.run_experiment --sample 4
"""

from __future__ import annotations

import argparse

from config import ExperimentConfig
from covariates.retriever import Retriever, StubRetriever
from data.loader import load_mmlongbench
from experiments.smoke import load_smoke_questions
from pipeline.conditioner import (
    BuriedOracle,
    FullDoc,
    InputConditioner,
    OracleConditioner,
    RetrievedTopK,
)
from pipeline.orchestrator import Orchestrator


def build_conditioners(config: ExperimentConfig, retriever: Retriever) -> list[InputConditioner]:
    """Expand the config's condition names + grids into conditioner instances."""

    conditioners: list[InputConditioner] = []
    for name in config.conditions:
        if name == "oracle":
            conditioners.append(OracleConditioner())
        elif name == "full":
            conditioners.append(FullDoc())
        elif name == "retrieved":
            conditioners += [RetrievedTopK(retriever, k) for k in config.k_values]
        elif name == "buried":
            conditioners += [BuriedOracle(n) for n in config.burying_levels]
        else:
            raise ValueError(f"unknown condition {name!r}")
    return conditioners


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the pipeline over a tiny sample.")
    parser.add_argument("--sample", type=int, default=4, help="number of questions")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the frozen MVP smoke corpus with the smoke model config",
    )
    args = parser.parse_args(argv)

    config = ExperimentConfig(smoke=args.smoke, sample=None if args.smoke else args.sample)
    if args.smoke:
        questions = load_smoke_questions(data_dir=config.paths.data_dir)
    else:
        questions = load_mmlongbench(data_dir=config.paths.data_dir, sample=args.sample)
    orchestrator = Orchestrator(config)
    conditioners = build_conditioners(config, StubRetriever())

    rows = []
    for question in questions:
        for conditioner in conditioners:
            for representation in config.representations:
                rows.append(orchestrator.run_cell(question, conditioner, representation))

    mode = "smoke" if config.smoke else "sample"
    print(f"ran {len(rows)} cells over {len(questions)} questions ({mode})")
    print(f"cache: {orchestrator.cache.path} ({len(orchestrator.cache)} rows)")
    correct = sum(1 for row in rows if row.correct)
    print(f"stub-correct: {correct}/{len(rows)}")


if __name__ == "__main__":
    main()
