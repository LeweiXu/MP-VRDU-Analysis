"""Paired within-question fidelity transitions: for questions answered at two rungs,
how often adding a channel flips the verdict, split by the evidence source cited or
by the native doc_type."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ._common import (
    Table,
    doc_type_of,
    group_by,
    ordered_doc_types,
    restrict_to_primary_spec,
    rows_for_condition,
)
from ._load import column_n_footer

# The pairings that matter. TL->TLV isolates the page image added on top of
# parser text (the "the parser dropped it, the image kept it" number); T->TL is the
# parser's own lift over the raw embedded text layer; T->TLV is the full lift.
PAIRINGS: tuple[tuple[str, str], ...] = (("TL", "TLV"), ("T", "TL"), ("T", "TLV"))
TRANSITIONS: tuple[tuple[str, bool, bool], ...] = (
    ("wrong→right (%)", False, True),
    ("right→wrong (%)", True, False),
    ("right→right (%)", True, True),
    ("wrong→wrong (%)", False, False),
)
ALL_SOURCES = "**All sources**"
ALL_DOC_TYPES = "**All doc_types**"
NO_SOURCE = "(none)"
NOTE = (
    "Paired on question_id over the same oracle pages: a question counts only when "
    "BOTH rungs produced a status==ok row, so the paired n is well below the pool and "
    "is the discount signal. "
    "The four transition columns are PERCENTAGES of that row's paired n and sum to 100 "
    "per row; the figure in parentheses is the raw question count behind the "
    "percentage. "
    "A question citing several evidence sources is counted under each of them, so the "
    "per-source rows overlap and CANNOT be summed. The bolded All sources row closing "
    "each block is therefore computed fresh over every paired question in that pairing, "
    "each counted once, with its own paired n; it is a pooled total, not a column sum."
)
DOCTYPE_NOTE = (
    "Paired on question_id over the same oracle pages: a question counts only when "
    "BOTH rungs produced a status==ok row, so the paired n is well below the pool and "
    "is the discount signal. "
    "The four transition columns are PERCENTAGES of that row's paired n and sum to 100 "
    "per row; the figure in parentheses is the raw question count behind the "
    "percentage. "
    "Each question carries exactly one doc_type, so unlike the evidence-source table "
    "the rows within a block are disjoint and the bolded All doc_types row closing it "
    "is their sum."
)


def _verdict_index(rows: Sequence[Any]) -> dict[tuple[str, str], Any]:
    """Map (question_id, rung) to its row. Rows are already ok-only and deduped."""

    return {(getattr(r, "question_id", ""), getattr(r, "representation", "")): r for r in rows}


def _sources_of(row: Any) -> tuple[str, ...]:
    """The evidence sources a question cites, bucketed to `(none)` when unlabelled."""

    sources = getattr(row, "evidence_sources", ()) or ()
    return tuple(str(s) for s in sources) or (NO_SOURCE,)


def _paired(index: dict[tuple[str, str], Any], from_rung: str, to_rung: str) -> list[tuple[Any, Any]]:
    """Every question with an ok row at both rungs, as (from_row, to_row)."""

    pairs = []
    for (qid, rung), row in index.items():
        if rung == from_rung:
            partner = index.get((qid, to_rung))
            if partner is not None:
                pairs.append((row, partner))
    return pairs


def _rate_row(label: list[str], pairs: Sequence[tuple[Any, Any]]) -> list[str]:
    """Transition rates for one bucket of pairs, plus its paired n."""

    total = len(pairs)
    cells = []
    for _, want_from, want_to in TRANSITIONS:
        hits = sum(
            1 for a, b in pairs
            if bool(getattr(a, "correct", False)) is want_from and bool(getattr(b, "correct", False)) is want_to
        )
        cells.append(f"{hits / total * 100:.1f} ({hits})" if total else "-")
    return [*label, *cells, str(total)]


def _blocks(
    rows: Sequence[Any], fanout: Callable[[Any], tuple[str, ...]]
) -> list[tuple[str, list[tuple[Any, Any]], dict[str, list]]]:
    """Per pairing: its label, all pairs, and the pairs fanned out by `fanout` keys
    (evidence sources, or the single doc_type)."""

    index = _verdict_index(restrict_to_primary_spec(rows_for_condition(rows, "oracle")))
    out = []
    for from_rung, to_rung in PAIRINGS:
        pairs = _paired(index, from_rung, to_rung)
        by_key: dict[str, list[tuple[Any, Any]]] = {}
        for pair in pairs:
            for key in fanout(pair[0]):
                by_key.setdefault(key, []).append(pair)
        out.append((f"{from_rung}->{to_rung}", pairs, by_key))
    return out


def _doc_type_key(row: Any) -> tuple[str, ...]:
    return (doc_type_of(row),)


def build(rows: Sequence[Any]) -> Table:
    """Verdict-transition rates per evidence source, for both rung pairings."""

    columns = ["pairing", "evidence_source", *[name for name, _, _ in TRANSITIONS], "paired n"]
    table_rows: list[list[str]] = []
    totals: list[str] = []

    for label, pairs, by_source in _blocks(rows, _sources_of):
        for source in sorted(by_source):
            table_rows.append(_rate_row([label, source], by_source[source]))
        # Closes the block. Computed over `pairs` (every paired question once), not by
        # summing the per-source rows above, which overlap on multi-source questions.
        table_rows.append(_rate_row([label, ALL_SOURCES], pairs))
        totals.append(f"{label}: {len(pairs)}")

    # Every row already carries its own paired n, so the footer states the paired
    # question count per pairing (the two block totals) rather than a per-column n.
    footer = column_n_footer(columns, {})
    footer[0][1] = ", ".join(totals) or "-"

    return Table(
        key="fidelity_transition",
        title="Fidelity: paired within-question verdict transitions by evidence source",
        columns=columns,
        rows=table_rows,
        note=NOTE,
        footer=footer,
    )


def build_by_doctype(rows: Sequence[Any]) -> Table:
    """Verdict-transition rates per native doc_type, for the rung pairings."""

    columns = ["pairing", "doc_type", *[name for name, _, _ in TRANSITIONS], "paired n"]
    table_rows: list[list[str]] = []
    totals: list[str] = []

    for label, pairs, by_type in _blocks(rows, _doc_type_key):
        for dt in ordered_doc_types([a for a, _ in pairs]):
            table_rows.append(_rate_row([label, dt], by_type[dt]))
        # Closes the block; doc_types are disjoint, so this one IS the column sum.
        table_rows.append(_rate_row([label, ALL_DOC_TYPES], pairs))
        totals.append(f"{label}: {len(pairs)}")

    footer = column_n_footer(columns, {})
    footer[0][1] = ", ".join(totals) or "-"

    return Table(
        key="fidelity_transition_doctype",
        title="Fidelity: paired within-question verdict transitions by doc_type",
        columns=columns,
        rows=table_rows,
        note=DOCTYPE_NOTE,
        footer=footer,
    )


def doctype_summary(rows: Sequence[Any]) -> Table:
    """Doc_type-collapsed view: one pooled transition-rate row per rung pairing."""

    columns = ["pairing", *[name for name, _, _ in TRANSITIONS], "paired n"]
    table_rows = [
        _rate_row([label], pairs) for label, pairs, _ in _blocks(rows, _doc_type_key)
    ]
    return Table(
        key="fidelity_transition_doctype_summary",
        title="Fidelity (overall): paired verdict transitions per rung pairing",
        columns=columns,
        rows=table_rows,
        footer=column_n_footer(columns, {}),
    )
