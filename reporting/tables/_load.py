"""Load and merge cached rows across run_tags for the table build.

Resolves each run_tag's cache path and concatenates its rows, which is how a table
draws on several run_tags at once (the digital+scanned scan-merge, or the
cross-run parser/scale/quant comparisons). Also builds the per-column n footer row.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from config import ExperimentConfig
from experiments.engine.paths import experiment_paths

from ._common import load_ok_rows, read_jsonl


def _paths(run_tag: str, task: str):
    return experiment_paths(ExperimentConfig(run_tag=run_tag), task)


def load_ok(run_tags: Sequence[str], task: str) -> list[Any]:
    """Judged `ok` rows for a task, concatenated across run_tags (missing tags skipped)."""

    rows: list[Any] = []
    for tag in run_tags:
        rows += load_ok_rows(_paths(tag, task).results)
    return rows


def load_predictions(run_tags: Sequence[str], task: str) -> list[Any]:
    """Raw prediction rows (all statuses, incl. oom/error) across run_tags."""

    rows: list[Any] = []
    for tag in run_tags:
        rows += read_jsonl(_paths(tag, task).predictions)
    return rows


def load_side(run_tags: Sequence[str], task: str, name: str) -> list[Any]:
    """Side-artifact rows (e.g. retrieval.jsonl, classifier.jsonl) across run_tags."""

    rows: list[Any] = []
    for tag in run_tags:
        rows += read_jsonl(_paths(tag, task).side_dir / name)
    return rows


def column_n_footer(columns: Sequence[str], n_by_col: Mapping[str, int]) -> list[list[str]]:
    """One footer row giving n per column: the first column labels it, metric
    columns show their count, other columns stay blank."""

    row: list[str] = []
    for i, col in enumerate(columns):
        if i == 0:
            row.append("n (per col)")
        elif col in n_by_col:
            row.append(str(n_by_col[col]))
        else:
            row.append("")
    return [row]
