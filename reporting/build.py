"""Assembles every analysis table from the build plan, captions each with the
baseline it holds fixed, and writes one CSV per table plus a combined all_tables.md."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from importlib import import_module
from inspect import signature
from pathlib import Path
from typing import Any

from config import BASELINE, ExperimentConfig
from reporting.plan import G3, PLAN, caption_for
from reporting.tables import _common as common
from reporting.tables import _load
from reporting.tables import _markdown as md
from reporting.tables._common import Table

log = logging.getLogger("mpvrdu.build")

# Fields that together identify one prediction (a cell without its judge). Rows
# sharing these belong to the same cell, so grouping on them collapses a
# multi-judge or re-run history down to one cell. Resolution is part of the cell
# identity (it changes the image the model sees), so two resolutions of the same
# cell stay distinct.
IDENTITY_FIELDS = ("question_id", "doc_id", "condition", "representation", "model_spec", "visual_resolution")


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


def _enrich_retrieval_rows(rows: Sequence[Any], config: Any) -> None:
    """Backfill `doc_type` and `dpi` on retrieval rows in place (older artifacts).

    Both fields were added to the retrieval side-artifact after some `retrieval.jsonl`
    files were written, so older rows leave them blank: `doc_type` would collapse the
    breakdown to `(unknown)`, and `dpi` would read 0. `doc_type` is a property of the
    document (recovered by doc_id from the corpus); `dpi` is the run's render setting
    (`config.dpi`), correct for a single-run build. Best-effort: if the corpus can't
    load, `doc_type` is left blank and the table still builds.
    """

    if not rows:
        return
    run_dpi = int(getattr(config, "dpi", 0) or 0)
    if run_dpi:
        for row in rows:
            if not _field(row, "dpi") and hasattr(row, "__dict__"):
                row.dpi = run_dpi
    if all(_field(row, "doc_type") for row in rows):
        return
    data_dir = getattr(getattr(config, "paths", None), "data_dir", None) or getattr(config, "data_dir", None)
    doc_type_by_id: dict[str, str] = {}
    from data.loader import load_longdocurl, load_mmlongbench

    for loader in (load_mmlongbench, load_longdocurl):
        try:
            for question in loader(data_dir):
                if question.doc_type:
                    doc_type_by_id.setdefault(question.doc_id, question.doc_type)
        except Exception as exc:  # noqa: BLE001 - one dataset being absent is fine
            log.debug("doc_type backfill: %s skipped (%s)", loader.__name__, exc)
    if not doc_type_by_id:
        log.warning("doc_type backfill: no corpus loaded, retrieval rows stay bucketed as unknown")
        return
    for row in rows:
        if not _field(row, "doc_type") and hasattr(row, "__dict__"):
            row.doc_type = doc_type_by_id.get(_field(row, "doc_id"), "")


def load_result_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a results jsonl file into a list of row dicts."""

    rows: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# -- plan-driven table assembly ---------------------------------------------


def _safe(build_fn, *args, **kwargs) -> Table | None:
    """Run one builder; a builder that raises is logged, not fatal."""

    try:
        return build_fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - one bad table must not sink the rest
        log.warning("table build failed (%s): %s", getattr(build_fn, "__name__", build_fn), exc)
        return None


def _resolve_builder(dotted: str):
    """Resolve a `"<module>.<fn>"` name to the builder in `reporting.tables`."""

    module_name, fn_name = dotted.split(".")
    return getattr(import_module(f"reporting.tables.{module_name}"), fn_name)


def _load_rows(entry) -> list[Any]:
    """Load the rows one plan entry reads (judged results, predictions, or a side file)."""

    if entry.reads == "predictions":
        return _load.load_predictions(entry.run_tags, entry.task)
    if entry.reads.startswith("side:"):
        name = entry.reads.split(":", 1)[1]
        rows = _load.load_side(entry.run_tags, entry.task, name)
        _enrich_retrieval_rows(rows, ExperimentConfig(run_tag=entry.run_tags[0]))
        return rows
    return _load.load_ok(entry.run_tags, entry.task)


def _build_entry(entry, *, margin_points: float) -> Table:
    """Load an entry's rows and run its builder (parser and routing are special-cased)."""

    builder = _resolve_builder(entry.builder)
    if entry.parser_by_tag:
        labeled = [(entry.parser_by_tag[tag], _load.load_ok((tag,), entry.task)) for tag in entry.run_tags]
        return builder(labeled, margin_points=margin_points)
    if entry.key == "routing":
        g1 = _load.load_ok(entry.run_tags, entry.task)
        classifier = _load.load_side((G3,), "G3_hallucination", "classifier.jsonl")
        return builder(g1, classifier, margin_points=margin_points)
    rows = _load_rows(entry)
    kwargs = {"margin_points": margin_points} if "margin_points" in signature(builder).parameters else {}
    return builder(rows, **kwargs)


def _build_summary(entry, *, margin_points: float) -> Table:
    """Build an entry's doc_type-collapsed summary from the same rows as its detail."""

    builder = _resolve_builder(entry.summary)
    if entry.parser_by_tag:
        labeled = [(entry.parser_by_tag[tag], _load.load_ok((tag,), entry.task)) for tag in entry.run_tags]
        return builder(labeled, margin_points=margin_points)
    rows = _load_rows(entry)
    kwargs = {"margin_points": margin_points} if "margin_points" in signature(builder).parameters else {}
    return builder(rows, **kwargs)


def assemble_from_plan(plan: Sequence[Any] = PLAN, *, margin_points: float = 3.0) -> list[Table]:
    """Build every table in the plan, attaching each one's baseline caption.

    A table whose source run_tags have no cache builds empty; a builder that raises
    is logged and skipped, so one bad table never sinks the rest. Entries with a
    `summary` also emit a doc_type-collapsed, markdown-only summary table.
    """

    tables: list[Table] = []
    for entry in plan:
        table = _safe(_build_entry, entry, margin_points=margin_points)
        if table is not None:
            table.key = entry.key
            table.caption = caption_for(entry)
            table.md = entry.detail_md
            tables.append(table)
        if entry.summary:
            summary = _safe(_build_summary, entry, margin_points=margin_points)
            if summary is not None:
                summary.key = f"{entry.key}_summary"
                summary.caption = {"view": "summary — pooled across all doc_types", **caption_for(entry)}
                summary.md = True
                summary.csv = False
                tables.append(summary)
    return tables


def baseline_preamble() -> str:
    """A short preamble naming the shared baseline every table's caption pins."""

    items = " · ".join(f"**{k}**: {v}" for k, v in BASELINE["G1_oracle_ladder"].items())
    return (
        "Every table changes ONE variable off the shared baseline below and holds the rest "
        "fixed; each caption states what it swept and what it pinned. G2 uses retrieved pages, "
        "G3 the unanswerable pool.\n\n> " + items
    )


def write_tables(tables: Sequence[Table], out_dir: str | Path, *, preamble: str = "") -> list[Path]:
    """Write each table to CSV plus one combined `all_tables.md`; return paths."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for table in tables:
        if not table.csv:
            continue
        csv_path = out / f"{table.key}.csv"
        common.write_csv(table, csv_path)
        written.append(csv_path)
    report = out / "all_tables.md"
    report.write_text(md.render_report([t for t in tables if t.md], preamble=preamble))
    written.append(report)
    return written
