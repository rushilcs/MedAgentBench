"""Task-type helpers for query vs action shaping."""

from __future__ import annotations


def task_type_from_id(task_id: str) -> int | None:
    for part in task_id.split("_"):
        if part.startswith("task"):
            try:
                return int(part[len("task"):])
            except ValueError:
                return None
    return None


# MedAgentBench: types that routinely require POST / state-changing actions.
ACTION_TASK_TYPES: frozenset[int] = frozenset({3, 5, 6, 8, 9, 10})


def is_action_family(task_id: str) -> bool:
    tt = task_type_from_id(task_id)
    return tt in ACTION_TASK_TYPES if tt is not None else False
