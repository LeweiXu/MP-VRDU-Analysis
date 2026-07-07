"""Build phase (local): aggregate judged rows into the eight paper tables.

Purpose:
    The thin local entry point over `experiments.reporting.build_tables`. It reads
    each generation task's judged `results.jsonl`, routes rows per the table ->
    source-task map, and writes the eight CSVs plus a combined `all_tables.md`.
    The CSVs are the source of truth; the markdown is for readability.

CLI:
    `python -m cli.build [--full] [--run-tag TAG] [options]`

Arguments:
    --full / --run-tag locate the cache (mode + namespace) to read. --output-dir
        overrides where CSVs land; --markdown sets the combined report path (or
        `none` to skip it). --bootstrap / --seed control the document-level
        bootstrap. No task selector: every table builds from its fixed source
        tasks. A table is written only once all its source tasks' generate+judge
        phases have finished (successful status + non-empty output); tables with
        unfinished or unimplemented dependencies are skipped, not stubbed, and any
        stale CSV from an earlier partial build is removed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.driver import config_from_args
from experiments.paths import configure_logging, mode
from experiments.reporting import build_tables


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="build from the full cache (default: smoke)")
    parser.add_argument("--run-tag", help="cache namespace to read; must match the generate/judge phases")
    parser.add_argument("--output-dir", type=Path, help="directory for table CSVs (default: results/tables/<mode>[-<run-tag>])")
    parser.add_argument("--markdown", help="combined markdown path (default: <output-dir>/all_tables.md; 'none' to skip)")
    parser.add_argument("--bootstrap", type=int, help="document-level bootstrap resamples (default: 200 smoke / 1000 full)")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    parser.add_argument("--verbose", action="store_true", help="DEBUG-level logging")
    parser.add_argument("--quiet", action="store_true", help="force INFO-level logging even for smoke")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    configure_logging(verbose=args.verbose or (config.smoke and not args.quiet))

    partition = mode(config) if config.run_tag is None else f"{mode(config)}-{config.run_tag}"
    output_dir = args.output_dir or (config.paths.results_dir / "tables" / partition)
    if args.markdown is None:
        markdown_path: Path | None = output_dir / "all_tables.md"
    elif str(args.markdown).lower() == "none":
        markdown_path = None
    else:
        markdown_path = Path(args.markdown)

    written = build_tables(
        config, output_dir, n_bootstrap=args.bootstrap, seed=args.seed, markdown_path=markdown_path
    )
    for key, path in written.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
