"""Shared helpers for the table builders: the Table container, row loading, bin
and rung ordering, and per-group accuracy/cost formatting."""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from config import DEFAULT_REASONER_SPEC
from scoring.accuracy import accuracy_summary
from scoring.cost import cost_summary
from scoring.frontier import RUNG_ORDER, FrontierCell, sufficiency_frontier

log = logging.getLogger("mpvrdu.build")

# The tables group by the native mmlongbench doc_type label. These are the seven
# classes the dataset ships; anything else (e.g. a longdocurl task tag) sorts after
# them, and a blank doc_type falls into the `(unknown)` bucket.
DOC_TYPE_ORDER: tuple[str, ...] = (
    "Academic paper",
    "Administration/Industry file",
    "Brochure",
    "Financial report",
    "Guidebook",
    "Research report / Introduction",
    "Tutorial/Workshop",
)
UNKNOWN_DOC_TYPE = "(unknown)"

# Identity of one cell (a prediction without its judge); used to collapse re-runs
# and multi-judge history to a single row per cell before aggregating. Resolution
# is part of it: two resolutions of the same cell are different cells.
IDENTITY_FIELDS = ("question_id", "doc_id", "condition", "representation", "model_spec", "visual_resolution")


def split_condition(cond: str) -> tuple[str, str]:
    """Split a condition into (base, prompt_mode).

    Conditions are `<base>__<prompt_mode>` (e.g. `oracle__none`,
    `retrieved_text_k3__targeted`). A condition with no `__` has no prompt mode.
    """

    base, sep, mode = (cond or "").partition("__")
    return (base, mode) if sep else (base, "")


def base_condition(cond: str) -> str:
    """The base of a condition, dropping the `__<prompt_mode>` suffix."""

    return split_condition(cond)[0]


@dataclass
class Table:
    """A built table: a key, title, column headers, string rows, an optional
    structured caption (config held fixed vs swept), and optional footer rows
    (e.g. the per-column n count). `csv`/`md` gate which outputs it lands in
    (doc_type-collapsed summary tables are markdown-only); `rq` is the research
    question the table answers, which is the section it lands under in the report."""

    key: str
    title: str
    columns: list[str]
    rows: list[list[str]]
    note: str = ""
    caption: dict[str, str] = field(default_factory=dict)
    footer: list[list[str]] = field(default_factory=list)
    csv: bool = True
    md: bool = True
    rq: str = ""


def as_row(data: Any) -> Any:
    """Expose a jsonl dict row through attribute access (what scoring expects)."""

    return SimpleNamespace(**data) if isinstance(data, dict) else data


def read_jsonl(path: str | Path) -> list[Any]:
    """Read a jsonl file into attribute-access row objects (empty if absent)."""

    p = Path(path)
    if not p.exists():
        return []
    return [as_row(json.loads(line)) for line in p.read_text().splitlines() if line.strip()]


def load_ok_rows(path: str | Path) -> list[Any]:
    """Read result rows, collapse each cell to one row, keep only `status == ok`.

    A failed-then-completed cell can appear twice; the `ok` row wins. Scoring
    ignores non-ok rows, so this is where they drop out of the tables.
    """

    best: dict[tuple, Any] = {}
    for row in read_jsonl(path):
        key = tuple(getattr(row, f, "") for f in IDENTITY_FIELDS)
        current = best.get(key)
        if current is None or (getattr(current, "status", "") != "ok" and getattr(row, "status", "") == "ok"):
            best[key] = row
    return [row for row in best.values() if getattr(row, "status", "") == "ok"]


def restrict_to_primary_spec(rows: Sequence[Any]) -> list[Any]:
    """Keep one reasoner when a cache pools several model_specs.

    The single-reasoner tables (headline/parser/resolution/composition/routing)
    group by doc_type/rung only, so a multi-spec run_tag (the reasoner or quant
    sweeps) would silently average 2B/4B/8B/32B into one accuracy cell. When more
    than one model_spec is present, keep the primary (`DEFAULT_REASONER_SPEC`), or
    the most common spec if the primary is absent, and warn. `scale` and the mined
    quant table deliberately skip this so they can compare specs.
    """

    rows = list(rows)
    specs = {getattr(r, "model_spec", "") for r in rows}
    if len(specs) <= 1:
        return rows
    counts: dict[str, int] = {}
    for r in rows:
        spec = getattr(r, "model_spec", "")
        counts[spec] = counts.get(spec, 0) + 1
    keep = DEFAULT_REASONER_SPEC if DEFAULT_REASONER_SPEC in specs else max(counts, key=counts.get)
    log.warning(
        "table build: %d model_specs pooled (%s); restricting to %r to avoid mixing reasoners",
        len(specs), ", ".join(sorted(specs)), keep,
    )
    return [r for r in rows if getattr(r, "model_spec", "") == keep]


def rows_for_condition(rows: Sequence[Any], base: str) -> list[Any]:
    """Rows whose base condition (the `__<prompt_mode>` suffix dropped) equals `base`.

    When nothing matches, warn and fall back to every row so the table still builds.
    That fallback used to be a silent `[... == base] or list(rows)` in each builder,
    which meant a condition-format drift (e.g. `oracle` becoming `oracle__none`) would
    quietly pool *every* condition into the "oracle" table instead of surfacing. The
    warning turns that class of aggregation bug from silent into visible; today the
    filter matches, so this changes no numbers.
    """

    rows = list(rows)
    kept = [r for r in rows if base_condition(getattr(r, "condition", "")) == base]
    if kept or not rows:
        return kept
    sample = getattr(rows[0], "condition", "")
    log.warning(
        "table build: no rows matched base condition %r (e.g. condition=%r); falling back to "
        "all %d rows — likely a condition-format drift, so this table may pool conditions",
        base, sample, len(rows),
    )
    return rows


def unanswerable_rows(rows: Sequence[Any]) -> list[Any]:
    """Rows flagged `is_unanswerable` (the G3 pool), with the same de-silenced fallback
    as `rows_for_condition`: warn and keep all rows if none are flagged."""

    rows = list(rows)
    kept = [r for r in rows if getattr(r, "is_unanswerable", False)]
    if kept or not rows:
        return kept
    log.warning(
        "table build: no rows flagged is_unanswerable; falling back to all %d rows — "
        "the source run_tag may not be the unanswerable pool", len(rows),
    )
    return rows


def prefill_ms(rows: Sequence[Any]) -> str:
    """Mean prefill latency in milliseconds (the decode-free, uncontaminated cost).

    Not every backend can measure it: the prefill/decode split comes from a streamer
    that times the first token, and a backend generating through a single blocking
    call (InternVL's `chat()`) records the end-to-end latency only and leaves the
    split at zero. Reporting that as `0` reads as "prefills instantly", the opposite
    of the truth, so a group that ran but measured no prefill renders as `-`.
    """

    rows = list(rows)
    if not rows:
        return "-"
    if not any(float(getattr(r, "prefill_latency_s", 0.0) or 0.0) > 0.0 for r in rows):
        return "-"
    return f"{cost_summary(rows).prefill_s * 1000:.0f}"


def doc_type_of(row: Any) -> str:
    """The row's native doc_type label, bucketed to `(unknown)` when blank."""

    return getattr(row, "doc_type", "") or UNKNOWN_DOC_TYPE


def ordered_doc_types(rows: Iterable[Any]) -> list[str]:
    """Present doc_types in the fixed order, any extras (then unknown) after."""

    present = {doc_type_of(row) for row in rows}
    ordered = [t for t in DOC_TYPE_ORDER if t in present]
    return ordered + sorted(present - set(ordered))


def group_by(rows: Iterable[Any], keyfn) -> dict[Any, list[Any]]:
    """Bucket rows by an arbitrary key function."""

    out: dict[Any, list[Any]] = {}
    for row in rows:
        out.setdefault(keyfn(row), []).append(row)
    return out


def acc_cell(rows: Sequence[Any]) -> str:
    """Format accuracy as `pct [ci_low-ci_high]`, or `-` for no rows."""

    rows = list(rows)
    if not rows:
        return "-"
    s = accuracy_summary(rows)
    return f"{s.accuracy * 100:.1f} [{s.ci_low * 100:.1f}-{s.ci_high * 100:.1f}]"


def frontier_rung(rows: Sequence[Any], *, margin_points: float = 3.0) -> str:
    """The cheapest sufficient rung across a group's rows (empty if none)."""

    cells: dict[str, FrontierCell] = {}
    for rung, group in group_by(rows, lambda r: getattr(r, "representation", "")).items():
        if rung in RUNG_ORDER and group:
            s = accuracy_summary(group)
            cells[rung] = FrontierCell(accuracy=s.accuracy, ci_high=s.ci_high)
    return sufficiency_frontier(cells, margin_points=margin_points) if cells else ""


def latency_ms(rows: Sequence[Any]) -> str:
    """Mean end-to-end latency in milliseconds for a group."""

    rows = list(rows)
    return f"{cost_summary(rows).latency_bs1_s * 1000:.0f}" if rows else "-"


# Every VRAM figure in the tables carries this caveat, so it lives in one place and
# the builders reference it rather than restating it.
SINGLE_DEVICE_VRAM_NOTE = (
    "⚠ VRAM is SINGLE-DEVICE and understates the true footprint. Cells were generated "
    "on 2x V100, and the reasoner loads with device_map=\"auto\", which shards the "
    "model across both GPUs for every spec (the shard is triggered by GPU count, not "
    "model size). But peak memory is recorded with `torch.cuda.max_memory_allocated()` "
    "and no device argument, so only device 0 is measured: reported minima land at "
    "about half each model's bf16 weight size (8B: 7.82 GB against ~16 GB of weights). "
    "Device 1's peak was never written to any row and is not recoverable from the "
    "cache. Treat these as a device-0 lower bound, not a deployment budget. "
    "See docs/CODEBASE_GUIDE.md Part B section 9."
)


def peak_vram_mb(rows: Sequence[Any]) -> str:
    """Peak VRAM in MB across a group, as measured on device 0 only.

    `cost_summary` takes the max over the group's rows, so this is the binding cell
    rather than an average. It is a lower bound on the real footprint: see
    `SINGLE_DEVICE_VRAM_NOTE`.
    """

    rows = list(rows)
    return f"{cost_summary(rows).peak_vram_bytes / 1e6:.0f}" if rows else "-"


def write_csv(table: Table, path: str | Path) -> None:
    """Write one table to CSV: caption comment rows, header, data rows, footer rows.

    The caption is written as leading `# <field>,<value>` rows so the held-fixed
    config travels with the CSV; the footer rows (e.g. per-column n) follow the data.
    """

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as handle:
        writer = csv.writer(handle)
        for key, value in table.caption.items():
            writer.writerow([f"# {key}", value])
        writer.writerow(table.columns)
        writer.writerows(table.rows)
        writer.writerows(table.footer)
