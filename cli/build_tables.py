"""Build paper-table CSVs from cached prediction and score rows.

Purpose:
    Turns a result-cache jsonl file into the eight Stage-M5 paper table shapes.
    This keeps table building separate from expensive model execution so cached
    runs can be re-aggregated locally.

Pipeline role:
    Reads `pipeline.orchestrator.ResultRow` records, delegates aggregation to
    `experiments.tables`, and writes CSVs under `results/tables/` by default.

CLI:
    `python -m cli.build_tables [options]`

Arguments:
    --cache PATH: result jsonl path (default:
        `results/cache/orchestrator/results.jsonl`).
    --output-dir PATH: directory for table CSVs (default: `results/tables`).
    --dataset NAME: dataset label for Table 4 (default: `mmlongbench`).
    --bootstrap N: document-level bootstrap resamples (default: 1000).
    --seed N: deterministic bootstrap seed (default: 0).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import ROOT
from experiments.tables import load_result_rows, write_all_tables


def build_parser() -> argparse.ArgumentParser:
    """Return the table-build CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "results" / "cache" / "orchestrator" / "results.jsonl",
        help="cached ResultRow jsonl file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "tables",
        help="directory for emitted table CSVs",
    )
    parser.add_argument("--dataset", default="mmlongbench", help="dataset label for Table 4")
    parser.add_argument("--bootstrap", type=int, default=1000, help="document-level bootstrap resamples")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build all table CSVs from a result cache."""

    args = build_parser().parse_args(argv)
    rows = load_result_rows(args.cache)
    paths = write_all_tables(
        rows,
        args.output_dir,
        dataset=args.dataset,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    print(f"loaded {len(rows)} rows from {args.cache}")
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
