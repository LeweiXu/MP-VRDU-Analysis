"""Movement 2: one row per inference-time lever, its baseline, its effect, and
the n behind both sides. Levers without data render as blank rows, not guesses."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

from scoring.accuracy import accuracy_summary

from ._common import Table, group_by, restrict_to_primary_spec, rows_for_condition, split_condition
from ._load import column_n_footer, load_ok

BLANK = "-"
NOTE = (
    "One row per lever: the measured baseline, the lever value, and the delta "
    "in points, each with its n. A blank row means the lever's run has not "
    "landed; a populated row is never imputed across pools. Metrics differ by "
    "row (accuracy at the named rung, or abstention rate on the unanswerable "
    "pool) and are named per row: read the metric column, not just the delta. "
    "The retrieval-depth row is PROVISIONAL (partial G2 pool). The abstention "
    "row reads the ORIGINAL three-mode G3 run (targeted vs none); its six-mode "
    "re-run replaces it when judged. Deltas across different pools/runs are "
    "directional findings, not matched comparisons."
)


def _acc(rows: Sequence[Any]) -> float | None:
    return accuracy_summary(list(rows)).accuracy if rows else None


def _oracle_rung(tags: tuple[str, ...], task: str, rung: str, *, spec: str | None = None) -> list[Any]:
    rows = rows_for_condition(load_ok(tags, task), "oracle")
    rows = [r for r in rows if getattr(r, "representation", "") == rung]
    if spec is not None:
        return [r for r in rows if getattr(r, "model_spec", "") == spec]
    return restrict_to_primary_spec(rows)


def _accuracy_pair(base: Sequence[Any], lever: Sequence[Any]) -> tuple[str, str, str, str] | None:
    if not base or not lever:
        return None
    b, v = _acc(base), _acc(lever)
    return (f"{b * 100:.1f}", f"{v * 100:.1f}", f"{(v - b) * 100:+.1f}", f"{len(base)}/{len(lever)}")


def _resolution_lever() -> tuple[str, str, str, str] | None:
    rows = load_ok(("g1-resolution-full", "g1-resolution-scanned"), "G1_oracle_ladder")
    tlv = [r for r in rows_for_condition(rows, "oracle") if getattr(r, "representation", "") == "TLV"]
    by_res = group_by(tlv, lambda r: getattr(r, "visual_resolution", ""))
    return _accuracy_pair(by_res.get("med", []), by_res.get("high", []))


def _interleaving_lever() -> tuple[str, str, str, str] | None:
    rows = rows_for_condition(load_ok(("g1-interleaved-tlv",), "G1_oracle_ladder"), "oracle")
    by_rep = group_by(rows, lambda r: getattr(r, "representation", ""))
    return _accuracy_pair(by_rep.get("TLV", []), by_rep.get("TLVi", []))


def _prompt_lever(mode: str, baseline_mode: str = "grounded") -> tuple[str, str, str, str] | None:
    rows = load_ok(("g4-faithfulness-full",), "G4_faithfulness_answerable")
    tlv = [r for r in rows if getattr(r, "representation", "") == "TLV"]
    by_mode = group_by(tlv, lambda r: split_condition(getattr(r, "condition", ""))[1])
    return _accuracy_pair(by_mode.get(baseline_mode, []), by_mode.get(mode, []))


def _abstention_lever() -> tuple[str, str, str, str] | None:
    rows = load_ok(("g3-hallucination-full",), "G3_hallucination")
    tlv = [r for r in rows if getattr(r, "representation", "") == "TLV"]
    by_mode = group_by(tlv, lambda r: split_condition(getattr(r, "condition", ""))[1])
    base, lever = by_mode.get("none", []), by_mode.get("targeted", [])
    if not base or not lever:
        return None
    rate = lambda group: sum(1 for r in group if getattr(r, "abstained", False)) / len(group)  # noqa: E731
    b, v = rate(base), rate(lever)
    return (f"{b * 100:.1f}", f"{v * 100:.1f}", f"{(v - b) * 100:+.1f}", f"{len(base)}/{len(lever)}")


def _kdepth_lever() -> tuple[str, str, str, str] | None:
    rows = restrict_to_primary_spec(load_ok(("g2-retrieval-full",), "G2_retrieval"))
    v_rung = [r for r in rows if getattr(r, "representation", "") == "V"]
    by_base = group_by(v_rung, lambda r: split_condition(getattr(r, "condition", ""))[0])
    return _accuracy_pair(by_base.get("retrieved_vision_k1", []), by_base.get("retrieved_vision_k5", []))


def _family_lever() -> tuple[str, str, str, str] | None:
    base = _oracle_rung(("g1-reasoner-full", "g1-reasoner-scanned"), "G1_oracle_ladder", "TLV",
                        spec="internvl3-8b-local")
    anchor = _oracle_rung(("g1-representation-full",), "G1_oracle_ladder", "TLV")
    return _accuracy_pair(anchor, base)


def _thinking_lever() -> tuple[str, str, str, str] | None:
    def ms(rows: Sequence[Any]) -> float | None:
        by_hop = group_by(rows, lambda r: getattr(r, "hop", ""))
        single, multi = by_hop.get("single", []), by_hop.get("multi", [])
        if not single or not multi:
            return None
        return accuracy_summary(multi).accuracy - accuracy_summary(single).accuracy

    base_rows = _oracle_rung(("g1-representation-full",), "G1_oracle_ladder", "TLV")
    lever_rows = _oracle_rung(("g1-reasoner-thinking",), "G1_oracle_ladder", "TLV",
                              spec="qwen3vl-8b-thinking-local")
    b, v = ms(base_rows), ms(lever_rows)
    if b is None or v is None:
        return None
    return (f"{b * 100:+.1f}", f"{v * 100:+.1f}", f"{(v - b) * 100:+.1f}",
            f"{len(base_rows)}/{len(lever_rows)}")


LEVERS: tuple[tuple[str, str, str, Callable[[], tuple[str, str, str, str] | None]], ...] = (
    ("resolution med→high", "E3 fidelity", "acc @ TLV", _resolution_lever),
    ("interleaving TLV→TLVi", "E4 reasoning", "acc @ TLV", _interleaving_lever),
    ("CoT prompt (grounded→cot)", "E4 reasoning", "acc @ TLV", lambda: _prompt_lever("cot")),
    ("extraction (grounded→extract_cot)", "E4 reasoning", "acc @ TLV", lambda: _prompt_lever("extract_cot")),
    ("abstention prompt (none→targeted)", "E5 faithfulness", "abstention @ TLV (unanswerable)", _abstention_lever),
    ("retrieval depth k1→k5 (vision)", "E2 selection", "acc @ V (PROVISIONAL)", _kdepth_lever),
    ("model swap 8B→InternVL3-8B", "reasoner", "acc @ TLV", _family_lever),
    ("thinking variant (M−S)", "E4 reasoning", "M−S @ TLV", _thinking_lever),
)


def build(rows: Sequence[Any]) -> Table:
    """One row per lever; sources are loaded per lever, `rows` is unused."""

    columns = ["lever", "targets", "metric", "baseline", "lever value", "delta", "n (base/lever)"]
    table_rows: list[list[str]] = []
    populated = 0
    for label, targets, metric, loader in LEVERS:
        cells = loader()
        if cells is None:
            table_rows.append([label, targets, metric, BLANK, BLANK, BLANK, BLANK])
        else:
            populated += 1
            table_rows.append([label, targets, metric, *cells])
    if not populated:
        raise ValueError("levers: no lever has any data")

    return Table(
        key="levers",
        title="Levers: what each inference-time intervention does, where data exists",
        columns=columns,
        rows=table_rows,
        note=NOTE,
        footer=column_n_footer(columns, {}),
    )
