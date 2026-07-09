"""Routes tasks to tables, assembles the routing table at build time, and writes CSV and markdown."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

# Fields that together identify one prediction (a cell without its judge). Rows
# sharing these belong to the same cell, so grouping on them collapses a
# multi-judge or re-run history down to one cell.
IDENTITY_FIELDS = ("question_id", "doc_id", "condition", "representation", "model_spec")


def _field(row: Any, name: str, default: Any = "") -> Any:
    """Read a field from a mapping row or an object row."""

    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def group_key(row: Any) -> tuple:
    """Return the prediction-identity key a row groups under."""

    return tuple(_field(row, name) for name in IDENTITY_FIELDS)


def group_rows(rows: Iterable[Any]) -> dict[tuple, list[Any]]:
    """Group result rows by prediction identity (one group per cell)."""

    groups: dict[tuple, list[Any]] = {}
    for row in rows:
        groups.setdefault(group_key(row), []).append(row)
    return groups


def load_result_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a results jsonl file into a list of row dicts."""

    rows: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# Which task(s) feed which content-named table. Routing is explicit so one task
# can feed several tables; the routing table itself is assembled at build time
# from G1's ladder rows plus the G4 classifier price. The per-table builders in
# `reporting.tables` consume the grouped rows.
TASK_TO_TABLES: Mapping[str, tuple[str, ...]] = {
    "G1_oracle_ladder": ("headline", "parser", "resolution", "scale", "composition", "routing"),
    "G2_retrieval": ("matched_cross", "kdepth", "retrieval_accuracy"),
    "G3_hallucination": ("hallucination",),
    "G4_classifier_pricing": ("routing",),
}


def tables_for_task(task_name: str) -> Sequence[str]:
    """Return the content-named tables a task feeds."""

    return TASK_TO_TABLES.get(task_name, ())


# -- build-time table assembly ----------------------------------------------

import logging  # noqa: E402

from reporting.tables import _common as common  # noqa: E402
from reporting.tables import _markdown as md  # noqa: E402
from reporting.tables import (  # noqa: E402
    composition,
    hallucination,
    headline,
    kdepth,
    matched_cross,
    parser as parser_table,
    resolution,
    retrieval_accuracy,
    routing,
    scale,
)
from reporting.tables._common import Table  # noqa: E402

log = logging.getLogger("mpvrdu.build")


def _safe(build_fn, *args, **kwargs) -> Table | None:
    """Run one builder; a builder that raises is logged, not fatal."""

    try:
        return build_fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - one bad table must not sink the rest
        log.warning("table build failed (%s): %s", getattr(build_fn, "__module__", build_fn), exc)
        return None


def assemble_tables(task_paths: Mapping[str, Any], *, config: Any = None, margin_points: float = 3.0) -> list[Table]:
    """Build every table its inputs are available for, across tasks.

    `task_paths` maps a task name to its `ExperimentPaths`. Routing is assembled
    once from G1's ladder rows plus the G4 classifier price; the rest read a single
    task's results or side-artifact. Missing inputs yield an empty table, not a
    crash.
    """

    def results(task: str) -> list[Any]:
        paths = task_paths.get(task)
        return common.load_ok_rows(paths.results) if paths else []

    def side(task: str, name: str) -> list[Any]:
        paths = task_paths.get(task)
        return common.read_jsonl(Path(paths.side_dir) / name) if paths else []

    parser_label = getattr(config, "parser_tool", "") if config else ""
    resolution_label = getattr(config, "visual_resolution", "") if config else ""

    g1 = results("G1_oracle_ladder")
    g2 = results("G2_retrieval")
    g2_retrieval = side("G2_retrieval", "retrieval.jsonl")
    g3 = results("G3_hallucination")
    g4_classifier = side("G4_classifier_pricing", "classifier.jsonl")

    candidates: list[Table | None] = []
    if g1:
        candidates += [
            _safe(headline.build, g1, margin_points=margin_points),
            _safe(parser_table.build, g1, parser_label=parser_label, margin_points=margin_points),
            _safe(resolution.build, g1, resolution_label=resolution_label, margin_points=margin_points),
            _safe(scale.build, g1),
            _safe(composition.build, g1),
            _safe(routing.build, g1, g4_classifier, margin_points=margin_points),
        ]
    if g2:
        candidates += [_safe(matched_cross.build, g2), _safe(kdepth.build, g2)]
    if g2_retrieval:
        candidates.append(_safe(retrieval_accuracy.build, g2_retrieval))
    if g3:
        candidates.append(_safe(hallucination.build, g3))
    return [t for t in candidates if t is not None]


def write_tables(tables: Sequence[Table], out_dir: str | Path) -> list[Path]:
    """Write each table to CSV plus one combined `all_tables.md`; return paths."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for table in tables:
        csv_path = out / f"{table.key}.csv"
        common.write_csv(table, csv_path)
        written.append(csv_path)
    report = out / "all_tables.md"
    report.write_text(md.render_report(tables))
    written.append(report)
    return written
