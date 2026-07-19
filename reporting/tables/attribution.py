"""Three-way error attribution per rung: how much accuracy is lost to representation,
to real retrieval, and what remains as an uncorrected reasoning upper bound."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from scoring.accuracy import accuracy_summary
from scoring.frontier import RUNG_ORDER

from ._common import Table, acc_cell, base_condition, group_by, restrict_to_primary_spec, rows_for_condition
from ._load import column_n_footer, load_ok

G2_RUN_TAG = "g2-retrieval-full"
G2_TASK = "G2_retrieval"
ARMS = ("text", "vision", "joint")

# Which retrieval cells are comparable at all. G2 only ever ran the TLV and V rungs,
# and its OOM attrition climbs steeply with k on TLV, so the deeper TLV cells survive
# only as a survivorship-biased easy subset. Anything outside this map is left blank
# rather than reported: see the note for why each cell is missing.
IN_SCOPE_K: dict[str, tuple[int, ...]] = {"V": (1, 3, 5), "TLV": (1, 3)}
BLANK = "-"

NOTE = (
    "PROVISIONAL (partial G2 pool). The retrieval-loss columns are built on the "
    "g2-retrieval-full generation pool, which was only ~36% pulled before the cluster "
    "migration with judging still in flight, so every retrieval number here is "
    "provisional until a clean re-run. "
    "The reasoning residual is a RAW, UNCORRECTED UPPER BOUND: it is simply the "
    "shortfall of the best oracle rung from 100%, and it still contains judge false "
    "negatives and answers that are correct without matching the gold span. Neither "
    "correction exists in the data, so neither has been netted out. Do not read the "
    "residual as a reasoning-error estimate. "
    "T and TL carry no retrieval column because G2 never ran the text-only rungs; "
    "TLV above k=3 and all k>=7 are blank because OOM attrition leaves too few "
    "comparable cells. Retrieval loss is measured against the BEST in-scope retrieval "
    "setting for that rung, so it is the conservative (smallest defensible) retrieval "
    "charge, and it is computed within-question against the same questions' oracle rows."
)


def _accuracy(rows: Sequence[Any]) -> float | None:
    return accuracy_summary(list(rows)).accuracy if rows else None


def _points(value: float | None) -> str:
    return f"{value * 100:+.1f}" if value is not None else BLANK


def _k_of(condition: str) -> int:
    """The k encoded in a `retrieved_<arm>_k<k>` base condition (0 if absent)."""

    tail = base_condition(condition).rsplit("_k", 1)
    return int(tail[1]) if len(tail) == 2 and tail[1].isdigit() else 0


def _arm_of(condition: str) -> str:
    for arm in ARMS:
        if base_condition(condition).startswith(f"retrieved_{arm}_"):
            return arm
    return ""


def _load_g2_rows() -> list[Any]:
    """G2 generation rows. Loaded here because the plan entry reads one task at a
    time and this table spans the oracle (G1) and retrieved (G2) tasks."""

    return restrict_to_primary_spec(load_ok((G2_RUN_TAG,), G2_TASK))


def _best_retrieval(
    g2_rows: Sequence[Any], oracle_by_q: dict[str, Any], rung: str
) -> tuple[str, float, float, int] | None:
    """The in-scope retrieval condition with the highest paired accuracy for a rung.

    Returns (label, retrieved accuracy, paired oracle accuracy, paired n). Both
    accuracies are computed over the SAME questions, so their difference is a clean
    within-question retrieval charge rather than a comparison of two different pools.
    """

    candidates = [r for r in g2_rows if getattr(r, "representation", "") == rung]
    best = None
    for condition, rows in group_by(candidates, lambda r: base_condition(getattr(r, "condition", ""))).items():
        arm, k = _arm_of(condition), _k_of(condition)
        if not arm or k not in IN_SCOPE_K.get(rung, ()):
            continue
        paired = [(r, oracle_by_q[qid]) for r in rows if (qid := getattr(r, "question_id", "")) in oracle_by_q]
        if not paired:
            continue
        retrieved_acc = _accuracy([r for r, _ in paired])
        oracle_acc = _accuracy([o for _, o in paired])
        if retrieved_acc is None or oracle_acc is None:
            continue
        if best is None or retrieved_acc > best[1]:
            best = (f"{arm} k{k}", retrieved_acc, oracle_acc, len(paired))
    return best


def build(rows: Sequence[Any]) -> Table:
    """Representation loss, retrieval loss, and the reasoning residual per rung."""

    oracle = restrict_to_primary_spec(rows_for_condition(rows, "oracle"))
    by_rung = group_by(oracle, lambda r: getattr(r, "representation", ""))
    present = [r for r in RUNG_ORDER if by_rung.get(r)]
    if not present:
        raise ValueError("attribution: no oracle rows, nothing to attribute")

    rung_acc = {rung: _accuracy(by_rung[rung]) for rung in present}
    best_rung = max(present, key=lambda r: rung_acc[r] or 0.0)
    best_acc = rung_acc[best_rung]

    g2_rows = _load_g2_rows()

    columns = [
        "rung", "oracle acc", "representation loss", "retrieval ref",
        "retrieved acc", "retrieval loss", "reasoning residual (raw UB)", "oracle n",
    ]
    table_rows: list[list[str]] = []
    for rung in present:
        by_q = {getattr(r, "question_id", ""): r for r in by_rung[rung]}
        best = _best_retrieval(g2_rows, by_q, rung)
        if best is None:
            ref, retrieved_cells = BLANK, [BLANK, BLANK]
        else:
            label, retrieved_acc, paired_oracle_acc, paired_n = best
            ref = f"{label} (paired n={paired_n})"
            retrieved_cells = [f"{retrieved_acc * 100:.1f}", _points(paired_oracle_acc - retrieved_acc)]
        table_rows.append([
            rung,
            acc_cell(by_rung[rung]),
            _points(best_acc - rung_acc[rung]) if rung != best_rung else "0.0 (best rung)",
            ref,
            *retrieved_cells,
            _points(1.0 - best_acc) if rung == best_rung else BLANK,
            str(len(by_rung[rung])),
        ])

    return Table(
        key="attribution",
        title="Attribution: representation / retrieval / reasoning loss per rung (PROVISIONAL)",
        columns=columns,
        rows=table_rows,
        note=NOTE,
        footer=column_n_footer(columns, {"oracle acc": len(oracle)}),
    )
