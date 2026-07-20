"""Load and merge cached rows across run_tags for the table build.

Resolves each run_tag's cache path and concatenates its rows, which is how a table
draws on several run_tags at once (the digital+scanned scan-merge, or the
cross-run parser/scale/quant comparisons). Also builds the per-column n footer row.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

from config import ROOT, ExperimentConfig
from experiments.engine.paths import experiment_paths

from ._common import load_ok_rows, read_jsonl


def _paths(run_tag: str, task: str):
    return experiment_paths(ExperimentConfig(run_tag=run_tag), task)


@lru_cache(maxsize=1)
def _scan_labels() -> dict[str, str]:
    """doc_id -> digital/scanned from annotations/auto_scan.csv (every doc is labelled)."""

    path = ROOT / "annotations" / "auto_scan.csv"
    if not path.exists():
        return {}
    with path.open() as handle:
        return {row["doc_id"]: row["auto_scan"] for row in csv.DictReader(handle)}


def _backfill_scan(rows: Sequence[Any]) -> list[Any]:
    """Fill a blank `scan_label` from auto_scan.csv so no doc reads as unlabelled.

    Some cells were generated before the auto-scan pass, so their rows carry an empty
    `scan_label`; auto_scan.csv labels every document, so recover it by doc_id.
    """

    labels = _scan_labels()
    for row in rows:
        if not getattr(row, "scan_label", "") and hasattr(row, "__dict__"):
            recovered = labels.get(getattr(row, "doc_id", ""))
            if recovered:
                row.scan_label = recovered
    return list(rows)


def load_ok(run_tags: Sequence[str], task: str) -> list[Any]:
    """Judged `ok` rows for a task, concatenated across run_tags (missing tags skipped)."""

    rows: list[Any] = []
    for tag in run_tags:
        rows += load_ok_rows(_paths(tag, task).results)
    return _backfill_scan(rows)


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


@lru_cache(maxsize=1)
def _weight_sizes() -> dict[str, tuple[int, str]]:
    """model_spec -> (weight bytes, how it was obtained), from annotations."""

    path = ROOT / "annotations" / "model_weights.csv"
    if not path.exists():
        return {}
    with path.open() as handle:
        return {row["model_spec"]: (int(row["weights_bytes"]), row["method"])
                for row in csv.DictReader(handle)}


def weights_mb(model_spec: str) -> str:
    """Weight-only memory in MB for a spec, `-` when the spec is not annotated.

    A static property of the checkpoint, so unlike the measured `peak_vram_bytes` it
    is complete rather than device-0 only, and it needs no re-run. Derived rows (the
    quantized variants) are marked with a trailing `~`. Regenerate the annotation with
    `ops/scripts/model_weight_sizes.py`.
    """

    entry = _weight_sizes().get(model_spec)
    if entry is None:
        return "-"
    total, method = entry
    return f"{total / 1e6:.0f}{'~' if method != 'exact' else ''}"


def column_n_footer(columns: Sequence[str], n_by_col: Mapping[str, int]) -> list[list[str]]:
    """One footer row giving n per column: the first column labels it, columns with a
    per-column count show it, and columns where n does not apply show `-`."""

    row: list[str] = []
    for i, col in enumerate(columns):
        if i == 0:
            row.append("n (per col)")
        elif col in n_by_col:
            row.append(str(n_by_col[col]))
        else:
            row.append("-")
    return [row]
