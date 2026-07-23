"""Selection: sufficiency (gold withheld by rank) and robustness (ranked
distractors added) from the page_set runs, read against the all-gold oracle
accuracy on the same pool."""

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

G1_RUN_TAG = "g1-representation-full"
G1_TASK = "G1_oracle_ladder"
PIVOT = "**all gold (G1 oracle, hop=multi)**"
NOTE = (
    "Rows group purely on the pageset condition grammar (ranking source, gold "
    "mode/count, distractor count); nothing is re-derived from page_indices. "
    "Sufficiency rows withhold or isolate ONE gold page by the named ranker's "
    "ordering; robustness rows hold the kept gold fixed and add ranked "
    "distractors, so read down a gold block for the dilution slope (valid "
    "within a gold count, not across: evidence and length both differ). The "
    "bolded pivot row is the same reasoner's plain oracle accuracy on the same "
    "answerable hop=multi pool, loaded from the G1 cache: every sufficiency "
    "condition reads against it. Per-cell n is load-bearing: OOM attrition is "
    "rung-dependent and the pool is hop=multi only."
)


def _sort_key(rule) -> tuple:
    return (rule.gold_mode, rule.gold_count, rule.distractor_count, rule.ranking_source)


def _condition_label(rule) -> str:
    gold = "all gold" if rule.gold_mode == "all" else f"{rule.gold_mode.replace('_', ' ')} {rule.gold_count}"
    return f"{gold} + {rule.distractor_count} distractors" if rule.distractor_count else gold


def _acc_cells(rows: Sequence[Any], rungs: Sequence[str]) -> list[str]:
    by_rung = group_by(rows, lambda r: getattr(r, "representation", ""))
    return [f"{acc_cell(by_rung.get(rung, []))} (n={len(by_rung.get(rung, []))})" for rung in rungs]


def _oracle_pivot(rungs: Sequence[str]) -> list[str] | None:
    """The all-gold oracle row on the same pool, loaded from the G1 cache."""

    g1 = restrict_to_primary_spec(rows_for_condition(load_ok((G1_RUN_TAG,), G1_TASK), "oracle"))
    multi = [r for r in g1 if getattr(r, "hop", "") == "multi"]
    if not multi:
        return None
    return [PIVOT, "-", *_acc_cells(multi, rungs), str(len(multi))]


def build(rows: Sequence[Any]) -> Table:
    """Selection conditions x ranking source x rung, sufficiency then robustness."""

    ruled = [(rule, r) for r in rows if (rule := pageset_rule(getattr(r, "condition", ""))) is not None]
    if not ruled:
        raise ValueError("selection: no pageset rows (G5 runs not generated/judged yet)")

    rungs = [r for r in RUNG_ORDER if any(getattr(row, "representation", "") == r for _, row in ruled)]
    columns = ["condition", "ranker", *rungs, "n"]

    sufficiency = [(rule, row) for rule, row in ruled if rule.distractor_count == 0 and rule.gold_mode != "all"]
    robustness = [(rule, row) for rule, row in ruled if rule.gold_mode == "all" or rule.distractor_count > 0
                  or (rule.gold_mode.startswith("keep_") and rule.distractor_count == 0
                      and any(other.distractor_count > 0 and other.gold_mode == rule.gold_mode
                              and other.gold_count == rule.gold_count for other, _ in ruled))]

    table_rows: list[list[str]] = []
    pivot = _oracle_pivot(rungs)
    if pivot is not None:
        table_rows.append(pivot)
    for block in (sufficiency, robustness):
        for (rule_key, ranker), pairs in sorted(
                group_by(block, lambda p: (_sort_key(p[0])[:3], p[0].ranking_source)).items()):
            block_rows = [row for _, row in pairs]
            rule = pairs[0][0]
            table_rows.append([_condition_label(rule), ranker, *_acc_cells(block_rows, rungs),
                               str(len(block_rows))])

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
