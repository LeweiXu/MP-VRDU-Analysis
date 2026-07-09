"""Render built tables to GitHub-flavored markdown (one table, or a report of
many)."""

from __future__ import annotations

from collections.abc import Sequence

from ._common import Table


def render_table(table: Table) -> str:
    """Render one table as a markdown section: title, optional note, grid."""

    lines = [f"### {table.title}", ""]
    if table.note:
        lines += [f"_{table.note}_", ""]
    header = "| " + " | ".join(table.columns) + " |"
    rule = "| " + " | ".join("---" for _ in table.columns) + " |"
    lines += [header, rule]
    if table.rows:
        for row in table.rows:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    else:
        lines.append("| " + " | ".join("" for _ in table.columns) + " |")
    lines.append("")
    return "\n".join(lines)


def render_report(tables: Sequence[Table]) -> str:
    """Render several tables into one markdown document."""

    return "# Tables\n\n" + "\n".join(render_table(t) for t in tables)
