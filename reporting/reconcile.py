"""Build-time reconciliation: anchor cells that must reproduce already-trusted
numbers, checked after assembly. A failing check withholds its table and fails
the build, because a silent mismatch is worse than a missing table."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# Trusted anchors, from the built all_tables.md (2026-07-23 tree, run_tag
# g1-representation-full / g3-hallucination-full, judge gemini-flash). These are
# properties of a specific cached dataset + judge, which is why they live here
# in the build rather than in tests (the test fixtures policy forbids pinning
# real-data numbers; the build is where the real data exists).
HEADLINE_LADDER = {"T": 31.9, "TL": 39.4, "TLV": 56.8, "V": 45.9}
# Pooled-across-rungs abstention rates from the hallucination table (the
# published anchor is the pooled rate, NOT a TLV-only cut).
G3_ABSTENTION = {"none": 16.9, "generic": 10.7, "targeted": 80.7}
# grounded/abstain are byte-identical instructions to generic/targeted, so the
# six-mode re-run must reproduce these rates; a material move means the prompt
# assembly drifted, not the finding.
G3F_ABSTENTION = {"none": 16.9, "grounded": 10.7, "abstain": 80.7}

_LEADING_FLOAT = re.compile(r"^[+-]?\d+(?:\.\d+)?")


def cell_value(cell: str) -> float | None:
    """The leading numeric figure of a table cell (`56.8 [52.0-61.2] (n=717)`)."""

    match = _LEADING_FLOAT.match(str(cell).strip())
    return float(match.group(0)) if match else None


def find_row(table: Any, **want: str) -> Sequence[str] | None:
    """The first row whose named columns hold the wanted values."""

    indices = {table.columns.index(col): value for col, value in want.items()}
    for row in table.rows:
        if all(str(row[i]) == value for i, value in indices.items()):
            return row
    return None


@dataclass(frozen=True)
class Check:
    """One anchor: a located cell must equal `expected` within `tol` points."""

    gates: str                                  # table key withheld on failure
    label: str
    expected: float
    locate: Callable[[Mapping[str, Any]], float | None]   # None = data absent -> skip
    tol: float = 0.05                           # both sides print at 1 dp


@dataclass(frozen=True)
class ReconcileResult:
    check: Check
    status: str                                 # pass | fail | skip
    actual: float | None = None


def _headline_cell(tables: Mapping[str, Any], rung: str) -> float | None:
    table = tables.get("headline_summary")
    if table is None or not table.rows:
        return None
    row = table.rows[0]
    return cell_value(row[table.columns.index(rung)]) if rung in table.columns else None


def _reasoner_8b_cell(tables: Mapping[str, Any], rung: str) -> float | None:
    table = tables.get("reasoner_unified")
    if table is None:
        return None
    row = find_row(table, block="precision", model_spec="qwen3vl-8b-local")
    return cell_value(row[table.columns.index(rung)]) if row is not None and rung in table.columns else None


def _faithfulness_none_cell(tables: Mapping[str, Any], rung: str) -> float | None:
    table = tables.get("faithfulness_pools")
    if table is None:
        return None
    row = find_row(table, prompt_mode="none", rung=rung)
    return cell_value(row[table.columns.index("answerable acc")]) if row is not None else None


def _g3f_abstention_cell(tables: Mapping[str, Any], mode: str) -> float | None:
    # The six-mode re-run pools into the same hallucination shape when its rows
    # land under the legacy builder; until a dedicated table exists, locate via
    # faithfulness_pools' pooled unanswerable column when present.
    table = tables.get("faithfulness_pools")
    if table is None:
        return None
    rows = [r for r in table.rows if str(r[0]) == mode]
    rates_n = []
    for row in rows:
        rate = cell_value(row[table.columns.index("unanswerable abstention (%)")])
        n_cell = str(row[table.columns.index("n (A/U)")])
        n = n_cell.rsplit("/", 1)[-1]
        if rate is not None and n.isdigit() and int(n) > 0:
            rates_n.append((rate, int(n)))
    if not rates_n:
        return None
    total = sum(n for _, n in rates_n)
    return sum(rate * n for rate, n in rates_n) / total


def _make_checks() -> tuple[Check, ...]:
    checks: list[Check] = []
    for rung, expected in HEADLINE_LADDER.items():
        checks.append(Check(
            gates="headline", label=f"headline {rung} reproduces the trusted ladder",
            expected=expected, locate=lambda t, r=rung: _headline_cell(t, r)))
        checks.append(Check(
            gates="reasoner_unified", label=f"reasoner_unified precision 8B {rung} == headline",
            expected=expected, locate=lambda t, r=rung: _reasoner_8b_cell(t, r)))
        # G4 none on oracle pages IS the headline ladder; valid because the G4
        # spec pins oracle (config.BASELINE records page_selection: oracle).
        checks.append(Check(
            gates="faithfulness_pools", label=f"G4 none {rung} reproduces the headline ladder",
            expected=expected, locate=lambda t, r=rung: _faithfulness_none_cell(t, r),
            tol=1.0))  # judge-time delimiter extraction may flip rare echo cells
    for mode, expected in G3F_ABSTENTION.items():
        checks.append(Check(
            gates="faithfulness_pools",
            label=f"G3 re-run {mode} abstention reproduces the legacy rate",
            expected=expected, locate=lambda t, m=mode: _g3f_abstention_cell(t, m),
            tol=3.0))  # rate-level: material moves mean assembly drift
    return tuple(checks)


CHECKS: tuple[Check, ...] = _make_checks()


def run_checks(tables: Sequence[Any], checks: Sequence[Check] = CHECKS) -> list[ReconcileResult]:
    """Evaluate every check against the built tables (keyed by table.key)."""

    by_key = {t.key: t for t in tables}
    results = []
    for check in checks:
        actual = check.locate(by_key)
        if actual is None:
            results.append(ReconcileResult(check, "skip"))
        elif abs(actual - check.expected) <= check.tol:
            results.append(ReconcileResult(check, "pass", actual))
        else:
            results.append(ReconcileResult(check, "fail", actual))
    return results


def failed_gates(results: Sequence[ReconcileResult]) -> set[str]:
    return {r.check.gates for r in results if r.status == "fail"}


def render_report(results: Sequence[ReconcileResult]) -> str:
    """A human-readable pass/fail/skip report."""

    lines = ["reconciliation:"]
    for r in results:
        detail = f" (expected {r.check.expected}, got {r.actual})" if r.status != "skip" else ""
        lines.append(f"  [{r.status.upper():4}] {r.check.label}{detail}")
    return "\n".join(lines)
