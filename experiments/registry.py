"""The experiment registry: names → `Experiment` instances, plus groups.

Purpose:
    One place that lists every paper-table experiment and the run order (T1 first
    because most tables reuse its rows). The driver and CLIs resolve `--experiment
    <name|group>` through here, so adding a table is a one-line registry change.

Pipeline role:
    Maps `experiments/T*_*.py` to stable names. `ALL` is the full ordered list;
    `GROUPS` gives RQ-level shortcuts. `resolve()` expands a selector to the
    experiments to run.

Arguments:
    None. Import-only; callers use `resolve(selector)`, `EXPERIMENTS`, `ORDER`.
"""

from __future__ import annotations

from collections.abc import Sequence

from experiments.base import Experiment
from experiments.T1_headline import Headline
from experiments.T2_analytical import Analytical
from experiments.T3_family import Family
from experiments.T4_dataset import Dataset
from experiments.T5_composition import Composition
from experiments.T6_matched_cross import MatchedCross
from experiments.T7_routing import Routing
from experiments.T8_scale import Scale


# Ordered so dependencies (T1) generate before the tables that reuse their rows.
_EXPERIMENTS: tuple[Experiment, ...] = (
    Headline(),
    Analytical(),
    Family(),
    Dataset(),
    Composition(),
    MatchedCross(),
    Routing(),
    Scale(),
)

EXPERIMENTS: dict[str, Experiment] = {exp.name: exp for exp in _EXPERIMENTS}
ORDER: tuple[str, ...] = tuple(exp.name for exp in _EXPERIMENTS)

GROUPS: dict[str, tuple[str, ...]] = {
    "all": ORDER,
    "rq1": ("T1_headline", "T2_analytical", "T3_family", "T4_dataset"),
    "rq2": ("T5_composition", "T6_matched_cross"),
    "rq3": ("T7_routing",),
    "appendix": ("T8_scale",),
}


def resolve(selector: str) -> list[Experiment]:
    """Expand a selector (an experiment name or a group) to experiments in order."""

    key = selector.strip()
    if key in GROUPS:
        names: Sequence[str] = GROUPS[key]
    elif key in EXPERIMENTS:
        names = (key,)
    else:
        raise ValueError(
            f"unknown experiment/group {selector!r}; choose from "
            f"{sorted(EXPERIMENTS)} or groups {sorted(GROUPS)}"
        )
    return [EXPERIMENTS[name] for name in names]
