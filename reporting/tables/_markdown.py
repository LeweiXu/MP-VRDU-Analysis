"""Render built tables to GitHub-flavored markdown (one table, or a report of
many)."""

from __future__ import annotations

from collections.abc import Sequence

from ._common import Table


def _cell(value: object) -> str:
    """A markdown table cell: escape `|` so joint names like `bm25|colqwen` don't
    split the row into extra columns."""

    return str(value).replace("|", "\\|")


def _grid_row(cells: Sequence[object]) -> str:
    return "| " + " | ".join(_cell(c) for c in cells) + " |"


def render_table(table: Table) -> str:
    """Render one table as a markdown section: title, caption, optional note, grid,
    footer rows."""

    lines = [f"### {table.title}", ""]
    if table.caption:
        lines += ["> " + " · ".join(f"**{k}**: {_cell(v)}" for k, v in table.caption.items()), ""]
    if table.note:
        lines += [f"_{table.note}_", ""]
    lines += [_grid_row(table.columns), "| " + " | ".join("---" for _ in table.columns) + " |"]
    if table.rows:
        lines += [_grid_row(row) for row in table.rows]
    else:
        lines.append("| " + " | ".join("" for _ in table.columns) + " |")
    lines += [_grid_row(foot) for foot in table.footer]
    lines.append("")
    return "\n".join(lines)


def render_report(tables: Sequence[Table], *, preamble: str = "") -> str:
    """Render several tables into one markdown document, with an optional preamble
    (e.g. the shared baseline every table's caption is measured against).

    Tables are grouped into research-question sections in `RQ_SECTIONS` order,
    keeping plan order within each section. A table whose `rq` matches no known
    section falls into the appendix rather than disappearing.
    """

    from reporting.plan import APPENDIX, RQ_SECTIONS

    head = "# Tables\n\n"
    if preamble:
        head += preamble.rstrip() + "\n\n"

    known = {rq for rq, _ in RQ_SECTIONS}
    by_rq: dict[str, list[Table]] = {rq: [] for rq, _ in RQ_SECTIONS}
    for table in tables:
        by_rq[table.rq if table.rq in known else APPENDIX].append(table)

    parts: list[str] = []
    for rq, heading in RQ_SECTIONS:
        section = by_rq[rq]
        if not section:
            continue
        parts.append(f"## {heading}\n")
        parts += [render_table(t) for t in section]
    return head + "\n".join(parts)
