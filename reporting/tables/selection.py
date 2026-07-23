"""Selection: sufficiency (gold withheld by rank) and robustness (all gold plus
ranked distractors, blocked by the question's gold count) from the page_set
runs, read against the same questions' oracle rows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from scoring.frontier import RUNG_ORDER

from ._common import (
    Table,
    acc_cell,
    group_by,
    pageset_rule,
    restrict_to_primary_spec,
    rows_for_condition,
)
from ._load import column_n_footer, load_ok
from .hop_rung import _evidence_page_counts

G1_RUN_TAG = "g1-representation-full"
G1_TASK = "G1_oracle_ladder"
NOTE = (
    "Rows group on the pageset condition grammar (ranking source, gold rule, "
    "distractor count); the robustness gold-count BLOCKS come from the corpus "
    "gold-page annotation (the +k design filters questions by exact gold count "
    "and feeds ALL their gold pages, so the block is a property of the "
    "question, not the rule). Each block's d=0 baseline is the bolded oracle "
    "row above it: all gold + no distractors IS the oracle condition, loaded "
    "from the G1 cache over the same questions, so the baseline is exact and "
    "was never re-generated. Read DOWN a block for the dilution slope at "
    "constant evidence; blocks are not comparable to each other (evidence and "
    "length both differ). Sufficiency rows withhold or isolate ONE gold page "
    "by the named ranker's ordering, on the hop=multi pool, and read against "
    "the bolded multi-pool oracle row. Per-cell n is load-bearing: OOM "
    "attrition is rung-dependent."
)


def _acc_cells(rows: Sequence[Any], rungs: Sequence[str]) -> list[str]:
    by_rung = group_by(rows, lambda r: getattr(r, "representation", ""))
    return [f"{acc_cell(by_rung.get(rung, []))} (n={len(by_rung.get(rung, []))})" for rung in rungs]


def _gold_count_of(row: Any) -> int:
    """The question's gold-page count, from the corpus annotation."""

    return _evidence_page_counts().get(getattr(row, "question_id", ""), 0)


def _g1_oracle_rows() -> list[Any]:
    return restrict_to_primary_spec(rows_for_condition(load_ok((G1_RUN_TAG,), G1_TASK), "oracle"))


def _oracle_row(label: str, oracle: Sequence[Any], keep, rungs: Sequence[str]) -> list[str] | None:
    """One bolded oracle baseline row over the questions `keep` selects."""

    rows = [r for r in oracle if keep(r)]
    if not rows:
        return None
    return [f"**{label}**", "-", *_acc_cells(rows, rungs), str(len(rows))]


def build(rows: Sequence[Any]) -> Table:
    """Selection conditions x ranking source x rung, sufficiency then robustness."""

    ruled = [(rule, r) for r in rows if (rule := pageset_rule(getattr(r, "condition", ""))) is not None]
    if not ruled:
        raise ValueError("selection: no pageset rows (G5 runs not generated/judged yet)")

    rungs = [r for r in RUNG_ORDER if any(getattr(row, "representation", "") == r for _, row in ruled)]
    columns = ["condition", "ranker", *rungs, "n"]
    oracle = _g1_oracle_rows()
    table_rows: list[list[str]] = []

    # -- sufficiency: gold withheld/isolated by rank, hop=multi pool ----------
    sufficiency = [(rule, row) for rule, row in ruled if rule.gold_mode != "all"]
    if sufficiency:
        pivot = _oracle_row("oracle (all gold, hop=multi)", oracle,
                            lambda r: getattr(r, "hop", "") == "multi", rungs)
        if pivot is not None:
            table_rows.append(pivot)
        for (mode, count, d, ranker), pairs in sorted(group_by(
                sufficiency,
                lambda p: (p[0].gold_mode, p[0].gold_count, p[0].distractor_count,
                           p[0].ranking_source)).items()):
            label = f"{mode.replace('_', ' ')} {count}"
            if d:
                label += f" + {d} distractors"
            table_rows.append([label, ranker,
                               *_acc_cells([row for _, row in pairs], rungs), str(len(pairs))])

    # -- robustness: all gold + k distractors, blocked by corpus gold count ---
    robustness = [(rule, row) for rule, row in ruled if rule.gold_mode == "all"]
    gold_counts = sorted({_gold_count_of(row) for _, row in robustness if _gold_count_of(row)})
    for n in gold_counts:
        baseline = _oracle_row(f"oracle (gold {n}, d=0)", oracle,
                               lambda r, n=n: _gold_count_of(r) == n, rungs)
        if baseline is not None:
            table_rows.append(baseline)
        block = [(rule, row) for rule, row in robustness if _gold_count_of(row) == n]
        for (d, ranker), pairs in sorted(group_by(
                block, lambda p: (p[0].distractor_count, p[0].ranking_source)).items()):
            table_rows.append([f"gold {n} + {d} distractors", ranker,
                               *_acc_cells([row for _, row in pairs], rungs), str(len(pairs))])

    n_by_col = {rung: sum(1 for _, row in ruled if getattr(row, "representation", "") == rung)
                for rung in rungs}
    return Table(
        key="selection",
        title="Selection: sufficiency and robustness under constructed page sets",
        columns=columns,
        rows=table_rows,
        note=NOTE,
        footer=column_n_footer(columns, n_by_col),
    )
