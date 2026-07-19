"""Paired within-question fidelity transitions: for questions answered at two rungs,
how often adding a channel flips the verdict, split by the evidence source cited."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._common import Table, group_by, restrict_to_primary_spec, rows_for_condition
from ._load import column_n_footer

# The two pairings that matter. TL->TLV isolates the page image added on top of
# parser text (the "the parser dropped it, the image kept it" number); T->TLV is the
# full lift over the raw embedded text layer.
PAIRINGS: tuple[tuple[str, str], ...] = (("TL", "TLV"), ("T", "TLV"))
TRANSITIONS: tuple[tuple[str, bool, bool], ...] = (
    ("wrong->right", False, True),
    ("right->wrong", True, False),
    ("right->right", True, True),
    ("wrong->wrong", False, False),
)
ALL_SOURCES = "(all sources)"
NO_SOURCE = "(none)"
NOTE = (
    "Paired on question_id over the same oracle pages: a question counts only when "
    "BOTH rungs produced a status==ok row, so the paired n is well below the pool and "
    "is the discount signal. Rates are percentages of that source's paired n and sum "
    "to 100 per row. A question citing several evidence sources is counted under each "
    "of them, so the per-source n sums above the paired total."
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


def _blocks(rows: Sequence[Any]) -> list[tuple[str, list[tuple[Any, Any]], dict[str, list]]]:
    """Per pairing: its label, all pairs, and the pairs fanned out by evidence source."""

    index = _verdict_index(restrict_to_primary_spec(rows_for_condition(rows, "oracle")))
    out = []
    for from_rung, to_rung in PAIRINGS:
        pairs = _paired(index, from_rung, to_rung)
        by_source: dict[str, list[tuple[Any, Any]]] = {}
        for pair in pairs:
            for source in _sources_of(pair[0]):
                by_source.setdefault(source, []).append(pair)
        out.append((f"{from_rung}->{to_rung}", pairs, by_source))
    return out


def build(rows: Sequence[Any]) -> Table:
    """Verdict-transition rates per evidence source, for both rung pairings."""

    columns = ["pairing", "evidence_source", *[name for name, _, _ in TRANSITIONS], "paired n"]
    table_rows: list[list[str]] = []
    totals: list[str] = []

    for label, pairs, by_source in _blocks(rows):
        table_rows.append(_rate_row([label, ALL_SOURCES], pairs))
        for source in sorted(by_source):
            table_rows.append(_rate_row([label, source], by_source[source]))
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
