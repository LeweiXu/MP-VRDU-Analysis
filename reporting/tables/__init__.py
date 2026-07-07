"""The paper table builders, one module per table, plus the aggregation entry point.

Purpose:
    Each `T*_*.py` module holds one table builder (`T1_headline` .. `T8_scale`),
    named to mirror the generation tasks (`G1_sufficiency` ..). `_common.py` holds
    the shared row/metric helpers; `_markdown.py` renders the two `.md` reports.
    This package `__init__` is the entry point: it re-exports every builder and the
    constants, and owns `build_all_tables` / `write_all_tables`, the aggregation
    that runs all eight builders and writes the CSVs + the combined markdown.

Pipeline role:
    `reporting.build` and the tests import builders and helpers from
    `reporting.tables` (this package), so the split is transparent to callers.
    Nothing here touches the GPU; it reads cached `ResultRow`s and writes CSV/MD.

Arguments:
    None. Import-only; callers pass result rows to `build_all_tables()` or call
    `write_all_tables()`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from config import DEFAULT_BINS
from pipeline.orchestrator import ResultRow

from ._common import (
    ROUTING_POLICIES,
    QUESTION_TYPES,
    TABLE_FILENAMES,
    TABLE_TITLES,
    analytical_question_type,
    load_result_rows,
)
from ._markdown import render_paper_tables_markdown, render_tables_markdown
from .T1_headline import build_table1_headline
from .T2_analytical import build_table2_analytical
from .T3_family import build_table3_family_replication
from .T4_dataset import build_table4_dataset_replication
from .T5_composition import build_table5_composition_mediation, predict_frontier_from_composition
from .T6_retrieval import build_table6_matched_vs_cross
from .T7_routing import build_table7_routing
from .T8_scale import build_table8_scale_sanity

__all__ = [
    "TABLE_FILENAMES",
    "TABLE_TITLES",
    "QUESTION_TYPES",
    "ROUTING_POLICIES",
    "load_result_rows",
    "analytical_question_type",
    "predict_frontier_from_composition",
    "build_table1_headline",
    "build_table2_analytical",
    "build_table3_family_replication",
    "build_table4_dataset_replication",
    "build_table5_composition_mediation",
    "build_table6_matched_vs_cross",
    "build_table7_routing",
    "build_table8_scale_sanity",
    "build_all_tables",
    "write_all_tables",
    "render_tables_markdown",
    "render_paper_tables_markdown",
]


def build_all_tables(
    rows: Sequence[ResultRow],
    *,
    dataset: str = "mmlongbench",
    bins: Sequence[str] = DEFAULT_BINS,
    margin_points: float = 3.0,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Build all eight paper table shapes."""

    return {
        "table1": build_table1_headline(
            rows, bins=bins, margin_points=margin_points, n_bootstrap=n_bootstrap, seed=seed
        ),
        "table2": build_table2_analytical(rows, bins=bins, n_bootstrap=n_bootstrap, seed=seed),
        "table3": build_table3_family_replication(
            rows, bins=bins, margin_points=margin_points, n_bootstrap=n_bootstrap, seed=seed
        ),
        "table4": build_table4_dataset_replication(
            rows, bins=bins, margin_points=margin_points, n_bootstrap=n_bootstrap, seed=seed
        ),
        "table5": build_table5_composition_mediation(
            rows, bins=bins, margin_points=margin_points, n_bootstrap=n_bootstrap, seed=seed
        ),
        "table6": build_table6_matched_vs_cross(
            rows, bins=bins, margin_points=margin_points, n_bootstrap=n_bootstrap, seed=seed
        ),
        "table7": build_table7_routing(
            rows, bins=bins, margin_points=margin_points, n_bootstrap=n_bootstrap, seed=seed
        ),
        "table8": build_table8_scale_sanity(
            rows, bins=bins, margin_points=margin_points, n_bootstrap=n_bootstrap, seed=seed
        ),
    }


def write_all_tables(
    rows: Sequence[ResultRow],
    output_dir: Path,
    *,
    dataset: str = "mmlongbench",
    bins: Sequence[str] = DEFAULT_BINS,
    margin_points: float = 3.0,
    n_bootstrap: int = 1000,
    seed: int = 0,
    markdown_path: Path | None = None,
    markdown_source: str | None = None,
) -> dict[str, Path]:
    """Write all eight table CSV files and return their paths.

    When `markdown_path` is set, also write a single markdown file with all eight
    tables filled in (blank skeletons for tables that have no cached rows).
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = build_all_tables(
        rows,
        dataset=dataset,
        bins=bins,
        margin_points=margin_points,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    paths: dict[str, Path] = {}
    for key, table in tables.items():
        path = output_dir / TABLE_FILENAMES[key]
        table.to_csv(path, index=False)
        paths[key] = path
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            render_tables_markdown(tables, source=markdown_source, n_rows=len(rows)) + "\n"
        )
        paths["markdown"] = markdown_path
    return paths
