"""Oracle accuracy per rung split by the questions' source dataset, which on the
current corpus resolves to a single stratum plus the uncoded OOM cells."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from scoring.frontier import RUNG_ORDER

from ._common import Table, acc_cell, group_by, restrict_to_primary_spec, rows_for_condition
from ._load import column_n_footer, load_predictions

G1_RUN_TAG = "g1-representation-full"
G1_TASK = "G1_oracle_ladder"
UNCODED = "(uncoded)"

NOTE = (
    "This table cannot separate inherited from native questions, and the blank is the "
    "finding. `metadata.source_dataset` is the loader's dataset identifier, not the "
    "upstream QA dataset a question came from, so every judged row reads `mmlongbench` "
    "and no inherited-minus-native gap column can be computed. MMLongBench-Doc does not "
    "publish per-question upstream provenance in the staged config (the parquet carries "
    "doc_id, doc_type, question, answer, evidence_pages, evidence_sources, "
    "answer_format and nothing else), and no annotation file supplies it. So the "
    "memorisation-suspect channel is UNMEASURABLE on current data, not measured and "
    "found absent; closing it needs hand-labelling of document origin, which no run "
    "spec can produce. "
    "The `(uncoded)` stratum is the cells whose metadata is empty because they OOMed "
    "before producing an answer; they are shown with their n and no accuracy rather "
    "than dropped or imputed."
)


def _stratum_of(row: Any) -> str:
    """The row's source dataset, bucketed to `(uncoded)` when metadata is empty."""

    metadata = getattr(row, "metadata", None) or {}
    return str(metadata.get("source_dataset", "") or UNCODED) if isinstance(metadata, dict) else UNCODED


def _uncoded_counts() -> dict[str, int]:
    """Per-rung count of cells with no source_dataset, from the prediction rows.

    The judged rows this builder is handed are `ok`-only, so the uncoded cells are
    invisible there. Counting them from predictions keeps the stratum's n live rather
    than pinned to a number that goes stale the moment a cell is recovered.
    """

    counts: dict[str, int] = {}
    for row in load_predictions((G1_RUN_TAG,), G1_TASK):
        if _stratum_of(row) == UNCODED:
            rung = getattr(row, "representation", "")
            counts[rung] = counts.get(rung, 0) + 1
    return counts


def build(rows: Sequence[Any]) -> Table:
    """Accuracy per rung for each source-dataset stratum, uncoded cells surfaced."""

    oracle = restrict_to_primary_spec(rows_for_condition(rows, "oracle"))
    present = [r for r in RUNG_ORDER if any(getattr(x, "representation", "") == r for x in oracle)]
    by_stratum = group_by(oracle, _stratum_of)

    columns = ["source_dataset", *present, "n"]
    table_rows: list[list[str]] = []
    for stratum in sorted(by_stratum):
        stratum_rows = by_stratum[stratum]
        by_rung = group_by(stratum_rows, lambda r: getattr(r, "representation", ""))
        cells = [acc_cell(by_rung.get(rung, [])) for rung in present]
        table_rows.append([stratum, *cells, str(len(stratum_rows))])

    uncoded = _uncoded_counts()
    if uncoded:
        table_rows.append([UNCODED, *[str(uncoded.get(rung, 0)) + " cells, no acc" for rung in present],
                           str(sum(uncoded.values()))])

    n_by_col = {rung: sum(1 for r in oracle if getattr(r, "representation", "") == rung) for rung in present}
    return Table(
        key="source_stratification",
        title="Source stratification: oracle accuracy by source dataset and rung",
        columns=columns,
        rows=table_rows,
        note=NOTE,
        footer=column_n_footer(columns, n_by_col),
    )
