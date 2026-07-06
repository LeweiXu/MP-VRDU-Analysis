"""The generation-task registry: names -> `GenerationTask`, plus groups.

Purpose:
    One place that lists every generation task and the run order. The driver and
    CLIs resolve `--generation <name|group>` through here, so adding a task is a
    one-line change (import the new `experiments/G*_*.py` class and append it).

Pipeline role:
    Maps `experiments/G*_*.py` to stable names. `GENERATION_TASKS` is the dict,
    `ORDER` the ordered names, `GROUPS` the shortcuts; `resolve()` expands a
    selector to the tasks to run.

Arguments:
    None. Import-only; callers use `resolve(selector)`, `GENERATION_TASKS`, `ORDER`.
"""

from __future__ import annotations

from experiments.base import GenerationTask
from experiments.G1_sufficiency import G1Sufficiency
from experiments.G2_family import G2Family
from experiments.G3_dataset import G3Dataset
from experiments.G5_retrieval import G5Retrieval
from experiments.G6_classifier import G6Classifier


# Ordered; a scale-sanity task (G4, feeding Table 8) is out of scope for now.
_TASKS: tuple[GenerationTask, ...] = (
    G1Sufficiency(),
    G2Family(),
    G3Dataset(),
    G5Retrieval(),
    G6Classifier(),
)

GENERATION_TASKS: dict[str, GenerationTask] = {task.name: task for task in _TASKS}
ORDER: tuple[str, ...] = tuple(task.name for task in _TASKS)

GROUPS: dict[str, tuple[str, ...]] = {
    "all": ORDER,
    # Just the reasoner-cell tasks (skip the classifier side task).
    "reasoners": ("G1_sufficiency", "G2_family", "G3_dataset", "G5_retrieval"),
}


def resolve(selector: str) -> list[GenerationTask]:
    """Expand a selector to generation tasks, in registry order, de-duplicated.

    A selector is a task name (`G1_sufficiency`), a group (`all`, `reasoners`),
    or a comma-separated list of either, so an ad-hoc subset runs as one job.
    """

    names: list[str] = []
    for token in selector.split(","):
        key = token.strip()
        if not key:
            continue
        if key in GROUPS:
            names.extend(GROUPS[key])
        elif key in GENERATION_TASKS:
            names.append(key)
        else:
            raise ValueError(
                f"unknown generation task/group {key!r}; choose from "
                f"{sorted(GENERATION_TASKS)} or groups {sorted(GROUPS)}"
            )
    ordered = [name for name in ORDER if name in set(names)]
    return [GENERATION_TASKS[name] for name in ordered]
