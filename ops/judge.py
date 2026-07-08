"""Judging entry point: scores cached predictions."""

from __future__ import annotations

import argparse

from config import ExperimentConfig
from experiments.engine.paths import configure_logging, experiment_paths, log
from experiments.registry import resolve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="all", help="task name or group")
    parser.add_argument("--judge-spec", default="stub")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    config = ExperimentConfig(judge_spec=args.judge_spec)
    from pipeline.orchestrator import PredictionCache

    # A generate pass already scores cells with its configured judge. This entry
    # re-scores cached predictions with another judge without re-running the
    # reasoner; it reports what is available per task.
    for task in resolve(args.task):
        paths = experiment_paths(config, task.name)
        predictions = PredictionCache(paths.predictions)
        log.info("judge %s: %d cached predictions (judge=%s)", task.name, len(predictions), config.judge_spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
