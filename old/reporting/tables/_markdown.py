"""Markdown rendering for the built tables (the two `.md` aggregations).

Purpose:
    Two report shapes over the built table DataFrames: `render_tables_markdown`
    (the full `all_tables.md`, every column, blank skeletons for tables with no
    rows) and `render_paper_tables_markdown` (the compact, paper-style
    `all_tables_summarised.md`: accuracy % + CI, frontier in bold).

Pipeline role:
    Called by the `reporting.tables` entry point (`write_all_tables`) and by
    `reporting.build`. Pure formatting; it reads DataFrames, writes strings.

Arguments:
    None. Import-only; the render functions take a {key: DataFrame} mapping.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import pandas as pd

from ._common import TABLE_FILENAMES, TABLE_TITLES


def _table_to_markdown(df: pd.DataFrame) -> str:
    """Render one table DataFrame as a GitHub markdown table.

    A non-empty table is rendered with rows (floats rounded for readability). An
    empty table still emits its column header plus one blank row, so the combined
    report shows the table's skeleton with blank fields instead of dropping it.
    """

    columns = [str(col) for col in df.columns]
    if not columns:
        return "_(no columns)_"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    if df.empty:
        blank = "| " + " | ".join("" for _ in columns) + " |"
        return "\n".join([header, separator, blank])
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].round(4)
    body = [
        "| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |"
        for row in formatted.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *body])


def render_tables_markdown(
    tables: Mapping[str, pd.DataFrame],
    *,
    source: str | None = None,
    n_rows: int | None = None,
) -> str:
    """Render all eight tables into a single markdown document.

    Tables with data are filled; tables with no matching rows keep their skeleton
    (column header + a blank row) so the report always shows all eight shapes.
    """

    from datetime import datetime, timezone

    lines: list[str] = ["# MP-VRDU results tables", ""]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    meta = [f"Generated {stamp}."]
    if source:
        meta.append(f"Source: `{source}`.")
    if n_rows is not None:
        meta.append(f"{n_rows} result rows.")
    lines.append(" ".join(meta))
    lines.append("")
    lines.append("Empty tables show a blank skeleton row: their experiment has no cached rows yet.")
    lines.append("")
    for key in TABLE_FILENAMES:
        number = key.removeprefix("table")
        df = tables.get(key, pd.DataFrame())
        lines.append(f"## Table {number} — {TABLE_TITLES.get(key, key)}")
        note = "no data yet" if df.empty else f"{len(df)} rows"
        lines.append(f"_CSV: `{TABLE_FILENAMES[key]}` ({note})_")
        lines.append("")
        lines.append(_table_to_markdown(df))
        lines.append("")
    return "\n".join(lines)


def _fmt_pct(value: object, digits: int = 1) -> str:
    """Format a 0-1 accuracy as a percentage string, or '' if missing/non-numeric."""

    try:
        return f"{float(value) * 100:.{digits}f}"
    except (TypeError, ValueError):
        return ""


def _acc_ci(record: Mapping[str, object], prefix: str) -> str:
    """'44.4 [35.0, 53.7]' for a rung prefix (T/TL/TLV/V), or '' if no data."""

    acc = _fmt_pct(record.get(f"{prefix}_acc"))
    if not acc or not record.get(f"{prefix}_n"):
        return ""
    return f"{acc} [{_fmt_pct(record.get(f'{prefix}_ci_low'))}, {_fmt_pct(record.get(f'{prefix}_ci_high'))}]"


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render a github-flavoured markdown table."""

    if not rows:
        return "_(no rows)_"
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join("" if cell is None else str(cell) for cell in row) + " |")
    return "\n".join(out)


def _paper_ladder(df: pd.DataFrame, id_cols: Sequence[str], id_headers: Sequence[str], *, extra: Sequence[str] = ()) -> str:
    """Compact rung table: id cols, n, T/TL/TLV/V (acc [CI], frontier bold), frontier + latency."""

    rungs = ("T", "TL", "TLV", "V")
    headers = [*id_headers, "n", *rungs, "Frontier"]
    has_latency = "latency_at_frontier_s" in df.columns
    if "frontier" in df.columns and has_latency:
        headers.append("Frontier lat (s)")
    headers.extend(extra)
    rows: list[list[object]] = []
    for _, r in df.iterrows():
        front = str(r.get("frontier", "")) if "frontier" in df.columns else ""
        cells = []
        for rung in rungs:
            cell = _acc_ci(r, rung)
            cells.append(f"**{cell}**" if cell and rung == front else cell)
        row: list[object] = [r[c] for c in id_cols] + [int(r.get("n_questions", 0) or 0), *cells, front]
        if "frontier" in df.columns and has_latency:
            row.append(f"{float(r.get('latency_at_frontier_s', 0) or 0):.2f}")
        row.extend(r.get(c, "") for c in extra)
        rows.append(row)
    return _md_table(headers, rows)


def _paper_table2(df: pd.DataFrame) -> str:
    rungs = ("T", "TL", "TLV", "V")
    headers = ["Bin", "Question type", "n", *rungs]
    rows = [
        [r["bin"], r["question_type"], int(r.get("n_questions", 0) or 0), *[_fmt_pct(r.get(f"{g}_acc")) for g in rungs]]
        for _, r in df.iterrows()
    ]
    return _md_table(headers, rows)


def _paper_table5(df: pd.DataFrame) -> str:
    headers = ["Bin", "Evidence", "Share %", "Modality frontier", "Bin frontier", "Predicted bin frontier", "Match"]
    rows = [
        [
            r["bin"], r["evidence_modality"], _fmt_pct(r.get("share")),
            r.get("modality_frontier", ""), r.get("bin_frontier", ""), r.get("predicted_bin_frontier", ""),
            "yes" if r.get("predicted_matches_bin") else "no",
        ]
        for _, r in df.iterrows()
    ]
    return _md_table(headers, rows)


def _paper_table6(df: pd.DataFrame) -> str:
    headers = ["Bin", "Pipeline", "Retrieval", "Accuracy [CI]", "Δ vs matched (pts)", "Retrieval F1"]
    rows = []
    for _, r in df.iterrows():
        acc = _fmt_pct(r.get("accuracy"))
        acc_ci = f"{acc} [{_fmt_pct(r.get('ci_low'))}, {_fmt_pct(r.get('ci_high'))}]" if acc else ""
        delta = r.get("delta_accuracy_vs_matched")
        delta_str = f"{float(delta) * 100:+.1f}" if delta is not None and str(delta) != "nan" else ""
        rows.append([r["bin"], r["pipeline"], r.get("retrieval_modality", ""), acc_ci, delta_str, _fmt_pct(r.get("retrieval_f1"))])
    return _md_table(headers, rows)


def _paper_table7(df: pd.DataFrame) -> str:
    headers = ["Policy", "Chosen rungs", "n", "Accuracy [CI]", "Total latency (s)"]
    rows = []
    for _, r in df.iterrows():
        acc = _fmt_pct(r.get("accuracy"))
        acc_ci = f"{acc} [{_fmt_pct(r.get('ci_low'))}, {_fmt_pct(r.get('ci_high'))}]" if acc else ""
        rows.append([r["policy"], r.get("chosen_rungs", ""), int(r.get("n_rows", 0) or 0), acc_ci, f"{float(r.get('total_latency_bs1_s', 0) or 0):.2f}"])
    return _md_table(headers, rows)


_PAPER_RENDERERS: Mapping[str, "Callable[[pd.DataFrame], str]"] = {
    "table1": lambda df: _paper_ladder(df, ["bin"], ["Bin"]),
    "table2": _paper_table2,
    "table3": lambda df: _paper_ladder(df, ["model_spec", "model_size", "bin"], ["Model", "Size", "Bin"], extra=["matches_primary_frontier"]),
    "table4": lambda df: _paper_ladder(df, ["dataset", "bin"], ["Dataset", "Bin"]),
    "table5": _paper_table5,
    "table6": _paper_table6,
    "table7": _paper_table7,
}


def render_paper_tables_markdown(tables: Mapping[str, pd.DataFrame], *, source: str | None = None) -> str:
    """Render the built tables as compact, paper-style markdown (all_tables_summarised.md).

    Only the interpretable columns: document-level accuracy as a percentage with a
    95% bootstrap CI in brackets, the frontier rung in bold. Tables that weren't
    built (unfinished dependency, or table 8's unimplemented scale task) get a
    one-line note instead of a wall of empty columns.
    """

    from datetime import datetime, timezone

    lines = ["# MP-VRDU results (paper tables)", ""]
    meta = [f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}."]
    if source:
        meta.append(f"Source: `{source}`.")
    lines.append(" ".join(meta))
    lines.append("")
    lines.append(
        "Cells are document-level accuracy (%) with a 95% bootstrap CI in [brackets]. "
        "Rungs: T = text, TL = text+layout, TLV = text+layout+vision, V = vision. "
        "The frontier (cheapest sufficient rung) is in **bold**."
    )
    lines.append("")
    for key in TABLE_FILENAMES:
        number = key.removeprefix("table")
        lines.append(f"## Table {number}. {TABLE_TITLES.get(key, key)}")
        lines.append("")
        df = tables.get(key)
        if df is None or df.empty:
            note = (
                "_Not built: scale task (G4) is not implemented._"
                if key == "table8"
                else "_Not built yet: its source experiments' generate/judge haven't all finished._"
            )
            lines.append(note)
        else:
            lines.append(_PAPER_RENDERERS[key](df))
        lines.append("")
    return "\n".join(lines)
