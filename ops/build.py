"""Table-build entry point: turns judged rows into tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import ExperimentConfig
from experiments.engine.paths import configure_logging, experiment_paths, log
from experiments.registry import resolve
from reporting.build import assemble_tables, write_tables


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="all", help="task name or group")
    parser.add_argument("--run-tag", default=None,
                        help="build from a run-tagged cache (results/cache/<run_tag>/…); "
                             "omit for the un-tagged cache")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    config = ExperimentConfig(run_tag=args.run_tag)

    task_paths = {}
    out_dir = None
    for task in resolve(args.task):
        paths = experiment_paths(config, task.name)
        out_dir = out_dir or paths.table_dir
        if paths.results.exists() or Path(paths.side_dir).exists():
            task_paths[task.name] = paths

    tables = assemble_tables(task_paths, config=config, margin_points=config.sufficiency_margin)
    if not tables:
        log.info("build: no results yet, nothing to assemble")
        return 0
    written = write_tables(tables, out_dir)
    for table in tables:
        log.info("build: %s (%d rows)", table.key, len(table.rows))
    log.info("build: wrote %d files to %s", len(written), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
