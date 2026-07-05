"""Build phase: aggregate judged rows into the paper tables, locally.

Purpose:
    The last role. Each of the eight tables is a pure aggregation over one or
    more generation tasks' judged rows (plus, for tables 6 and 7, a side
    artifact). This module owns that routing: which task(s) feed which table.
    Because the builders in `experiments/tables.py` mostly don't filter by
    model_spec, handing each table exactly its source tasks' rows is what keeps
    them correct (this replaces the old per-experiment `depends_on`).

Pipeline role:
    Reads `results/cache/<mode>[/<run_tag>]/<task>/results.jsonl` (written by
    `experiments/judge.py`) and each side artifact, writes the eight CSVs under
    `results/tables/...` and one combined markdown for readability. No GPU, no
    judge — just pandas.

CLI:
    `python -m experiments.build [--full] [--run-tag TAG] [options]`

Arguments:
    --full / --run-tag locate the cache (mode + namespace) to read. --output-dir
        overrides where CSVs land; --markdown sets the combined report path (or
        `none` to skip it). --bootstrap / --seed control the document-level
        bootstrap. No task selector: every table builds from its fixed source
        tasks, with a blank skeleton for any table whose source has no rows.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from config import ExperimentConfig
from experiments.generation import GENERATION_TASKS, config_from_args
from experiments.paths import configure_logging, experiment_paths, log, mode
from experiments.tables import (
    TABLE_FILENAMES,
    build_table1_headline,
    build_table2_analytical,
    build_table3_family_replication,
    build_table4_dataset_replication,
    build_table5_composition_mediation,
    build_table6_matched_vs_cross,
    build_table7_routing,
    build_table8_scale_sanity,
    load_result_rows,
    render_tables_markdown,
)
from pipeline.orchestrator import ResultRow


def bootstrap_resamples(config: ExperimentConfig) -> int:
    """Document-level bootstrap resamples: fewer for smoke, 1000 for full."""

    return 200 if config.smoke else 1000


# A builder takes (rows, side_records, config, n_bootstrap, seed) -> DataFrame.
# `side_records` maps a source-task name to its loaded side-artifact records.
Builder = Callable[[Sequence[ResultRow], Mapping[str, list], ExperimentConfig, int, int], pd.DataFrame]


@dataclass(frozen=True)
class TableSpec:
    """One table: which task rows and side artifacts feed it, and how to build."""

    key: str
    sources: tuple[str, ...]          # generation tasks whose judged rows feed it
    build: Builder
    side_sources: tuple[str, ...] = ()  # generation tasks whose side artifact feeds it


def _t1(rows, side, c, nb, seed):
    return build_table1_headline(rows, bins=c.bins, margin_points=c.sufficiency_margin, n_bootstrap=nb, seed=seed)


def _t2(rows, side, c, nb, seed):
    return build_table2_analytical(rows, bins=c.bins, n_bootstrap=nb, seed=seed)


def _t3(rows, side, c, nb, seed):
    return build_table3_family_replication(rows, bins=c.bins, margin_points=c.sufficiency_margin, n_bootstrap=nb, seed=seed)


def _t4(rows, side, c, nb, seed):
    return build_table4_dataset_replication(rows, bins=c.bins, margin_points=c.sufficiency_margin, n_bootstrap=nb, seed=seed)


def _t5(rows, side, c, nb, seed):
    return build_table5_composition_mediation(rows, bins=c.bins, margin_points=c.sufficiency_margin, n_bootstrap=nb, seed=seed)


def _t6(rows, side, c, nb, seed):
    return build_table6_matched_vs_cross(
        rows, bins=c.bins, margin_points=c.sufficiency_margin,
        retrieval_records=side.get("G5_retrieval", []), n_bootstrap=nb, seed=seed,
    )


def _t7(rows, side, c, nb, seed):
    return build_table7_routing(
        rows, bins=c.bins, margin_points=c.sufficiency_margin,
        classifier_records=side.get("G6_classifier", []), n_bootstrap=nb, seed=seed,
    )


def _t8(rows, side, c, nb, seed):
    return build_table8_scale_sanity(rows, bins=c.bins, margin_points=c.sufficiency_margin, n_bootstrap=nb, seed=seed)


# The table -> source-task routing. G4 (scale) is out of scope, so table8 sources
# only G1 for now and shows the single primary size.
TABLES: tuple[TableSpec, ...] = (
    TableSpec("table1", ("G1_sufficiency",), _t1),
    TableSpec("table2", ("G1_sufficiency",), _t2),
    TableSpec("table3", ("G1_sufficiency", "G2_family"), _t3),
    TableSpec("table4", ("G3_dataset",), _t4),
    TableSpec("table5", ("G1_sufficiency",), _t5),
    TableSpec("table6", ("G5_retrieval",), _t6, side_sources=("G5_retrieval",)),
    TableSpec("table7", ("G1_sufficiency",), _t7, side_sources=("G6_classifier",)),
    TableSpec("table8", ("G1_sufficiency",), _t8),
)


def _load_side_records(path: Path) -> list[dict]:
    """Load a side-artifact jsonl (retrieval/classifier), empty if absent."""

    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_tables(
    config: ExperimentConfig,
    output_dir: Path,
    *,
    n_bootstrap: int | None = None,
    seed: int = 0,
    markdown_path: Path | None = None,
) -> dict[str, Path]:
    """Route each table's source-task rows + side artifacts and write CSVs (+ MD)."""

    n_bootstrap = bootstrap_resamples(config) if n_bootstrap is None else n_bootstrap
    output_dir.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}
    written: dict[str, Path] = {}
    sources_used: set[str] = set()

    for spec in TABLES:
        rows: list[ResultRow] = []
        for task_name in spec.sources:
            results_path = experiment_paths(config, task_name).results
            task_rows = list(load_result_rows(results_path))
            if task_rows:
                sources_used.add(task_name)
            rows.extend(task_rows)
        side: dict[str, list] = {}
        for task_name in spec.side_sources:
            artifact = GENERATION_TASKS[task_name].side_artifact
            if artifact:
                side[task_name] = _load_side_records(experiment_paths(config, task_name).side_dir / artifact)
        frame = spec.build(rows, side, config, n_bootstrap, seed)
        tables[spec.key] = frame
        out = output_dir / TABLE_FILENAMES[spec.key]
        frame.to_csv(out, index=False)
        written[spec.key] = out
        log.info("built %s (%d rows) from %s -> %s", spec.key, len(frame), ",".join(spec.sources), out)

    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        source_label = ", ".join(sorted(sources_used)) or "(no cached rows)"
        markdown_path.write_text(
            render_tables_markdown(tables, source=source_label, n_rows=None) + "\n"
        )
        written["markdown"] = markdown_path
    return written


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
    # Only the cache-locating knobs matter for building (mode + run-tag).
    args_ns = argparse.Namespace(
        full=args.full, questions=None, per_bin_questions=None, sample_seed=None,
        quantization=None, visual_resolution=None, run_tag=args.run_tag,
    )
    config = config_from_args(args_ns)
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
