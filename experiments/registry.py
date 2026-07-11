"""Resolves a task_name label to the single spec-driven generation task.

Task names are just labels (cache namespace + parallel job); every one runs the
same `Task`, whose behavior comes from the config.
"""

from __future__ import annotations

from experiments.tasks.base import GenerationTask
from experiments.tasks.task import Task

# The canonical experiment labels. Any label is valid (it only names a cache dir);
# these are the ones the groups below expand to.
CANONICAL: tuple[str, ...] = ("G1_oracle_ladder", "G2_retrieval", "G3_hallucination")

GROUPS: dict[str, tuple[str, ...]] = {
    "all": CANONICAL,
    "reasoners": CANONICAL,
}

# The canonical labels mapped to their task, for selectors/tooling that enumerate
# them. Any other label is still valid; it just names a different cache dir.
TASKS: dict[str, GenerationTask] = {name: Task(name) for name in CANONICAL}


def get_task(name: str) -> GenerationTask:
    """Return the spec-driven task under a name (the name is its cache namespace)."""

    return Task(name)


def resolve(selector: str) -> list[GenerationTask]:
    """Expand a selector (task_name label, a group, or a comma-separated list) to tasks."""

    names: list[str] = []
    for token in selector.split(","):
        key = token.strip()
        if not key:
            continue
        if key in GROUPS:
            names.extend(GROUPS[key])
        else:
            names.append(key)
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return [Task(name) for name in ordered]
