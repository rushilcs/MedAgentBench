"""Soft curriculum: skew query vs action sampling without hard phase splits (plan §5, §9)."""

from __future__ import annotations

import random
from typing import Any

from rl_training.rl.verifiers.task_masks import is_action_family


def apply_soft_curriculum_mix(
    tasks: list[dict[str, Any]],
    query_target_fraction: float = 0.7,
    *,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Resample tasks with replacement to target ``query_target_fraction`` query tasks.

    Action tasks are those whose ``id`` matches ``is_action_family``. Length is
    preserved so GRPO batching math stays unchanged.
    """
    if not tasks:
        return tasks
    q = [t for t in tasks if not is_action_family(t.get("id", ""))]
    a = [t for t in tasks if is_action_family(t.get("id", ""))]
    if not q:
        return list(tasks)
    if not a:
        return list(tasks)
    rng = random.Random(seed)
    n = len(tasks)
    out: list[dict[str, Any]] = []
    for _ in range(n):
        if rng.random() < query_target_fraction:
            out.append(rng.choice(q))
        else:
            out.append(rng.choice(a))
    return out
