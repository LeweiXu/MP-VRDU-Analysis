"""Table-build entry point: turns judged rows into tables."""

from __future__ import annotations

import argparse

from config import ExperimentConfig
from experiments.engine.paths import configure_logging, experiment_paths, log
from experiments.registry import resolve
from reporting.build import group_rows, load_result_rows, tables_for_task


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="all", help="task name or group")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    config = ExperimentConfig()
    for task in resolve(args.task):
        paths = experiment_paths(config, task.name)
        if not paths.results.exists():
            log.info("build %s: no results.jsonl yet, skipping", task.name)
            continue
        rows = load_result_rows(paths.results)
        groups = group_rows(rows)
        log.info("build %s: %d rows -> %d cells; feeds tables %s",
                 task.name, len(rows), len(groups), list(tables_for_task(task.name)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
