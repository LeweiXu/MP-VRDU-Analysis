"""Table-build entry point: assembles every analysis table from the build plan,
runs the reconciliation checks, and writes results/tables (one CSV each plus
all_tables.md). A failed reconciliation withholds the gated table and fails the
build: a silent mismatch is worse than a missing table."""

from __future__ import annotations

import argparse

from config import ROOT, ExperimentConfig
from experiments.engine.paths import configure_logging, log
from reporting.build import assemble_from_plan, baseline_preamble, generation_report, write_tables
from reporting.plan import PLAN
from reporting.reconcile import failed_gates, render_report, run_checks


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

    # Reconciliation gates outputs; _safe only covers builder crashes. Skips
    # (source data absent) are reported and non-fatal; fails withhold the gated
    # table (and its summary) and fail the build.
    results = run_checks(tables)
    gated = failed_gates(results)
    if any(r.status != "pass" for r in results) or gated:
        log.info("%s", render_report(results))
    preamble = generation_report() + "\n\n" + baseline_preamble()
    if gated:
        log.error("reconciliation FAILED; withholding: %s", sorted(gated))
        tables = [t for t in tables if t.key not in gated and t.key.removesuffix("_summary") not in gated]
        preamble = "## ⚠ RECONCILIATION FAILED\n\n```\n" + render_report(results) + "\n```\n\n" + preamble

    out_dir = ROOT / "results" / "tables"
    written = write_tables(tables, out_dir, preamble=preamble)
    for table in tables:
        log.info("build: %s (%d rows)", table.key, len(table.rows))
    log.info("build: wrote %d tables + all_tables.md to %s", len(tables), out_dir)
    if gated:
        return 1
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
