"""Maps a task name to its task collection."""

from __future__ import annotations

from experiments.tasks.base import GenerationTask
from experiments.tasks.G1_oracle_ladder import G1OracleLadder
from experiments.tasks.G2_retrieval import G2Retrieval
from experiments.tasks.G3_hallucination import G3Hallucination
from experiments.tasks.G4_classifier_pricing import G4ClassifierPricing


_TASKS: tuple[GenerationTask, ...] = (
    G1OracleLadder(),
    G2Retrieval(),
    G3Hallucination(),
    G4ClassifierPricing(),
)

TASKS: dict[str, GenerationTask] = {task.name: task for task in _TASKS}
ORDER: tuple[str, ...] = tuple(task.name for task in _TASKS)

GROUPS: dict[str, tuple[str, ...]] = {
    "all": ORDER,
    # Just the reasoner-cell tasks (skip the side-only classifier pricing).
    "reasoners": ("G1_oracle_ladder", "G2_retrieval", "G3_hallucination"),
}


def get_task(name: str) -> GenerationTask | None:
    """Return the task registered under a name, or None."""

    return TASKS.get(name)


def resolve(selector: str) -> list[GenerationTask]:
    """Expand a selector to generation tasks, in registry order, de-duplicated.

    A selector is a task name, a group (`all`, `reasoners`), or a comma-separated
    list of either, so an ad-hoc subset runs as one job.
    """

    names: list[str] = []
    for token in selector.split(","):
        key = token.strip()
        if not key:
            continue
        if key in GROUPS:
            names.extend(GROUPS[key])
        elif key in TASKS:
            names.append(key)
        else:
            raise ValueError(
                f"unknown generation task/group {key!r}; choose from {sorted(TASKS)} or groups {sorted(GROUPS)}"
            )
    ordered = [name for name in ORDER if name in set(names)]
    return [TASKS[name] for name in ordered]
