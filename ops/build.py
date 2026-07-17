"""Table-build entry point: assembles every analysis table from the build plan and
writes them to results/tables (one CSV each plus all_tables.md)."""

from __future__ import annotations

import argparse

from config import ROOT, ExperimentConfig
from experiments.engine.paths import configure_logging, log
from reporting.build import assemble_from_plan, baseline_preamble, write_tables
from reporting.plan import PLAN


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None, help="build only the table with this key")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    plan = tuple(e for e in PLAN if e.key == args.only) if args.only else PLAN
    tables = assemble_from_plan(plan, margin_points=ExperimentConfig().sufficiency_margin)
    out_dir = ROOT / "results" / "tables"
    written = write_tables(tables, out_dir, preamble=baseline_preamble())
    for table in tables:
        log.info("build: %s (%d rows)", table.key, len(table.rows))
    log.info("build: wrote %d tables + all_tables.md to %s", len(tables), out_dir)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
