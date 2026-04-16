from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rl_training.data.trajectory import Trajectory

QUERY_TASK_TYPES = {"task1", "task2", "task4", "task6", "task7"}
ACTION_TASK_TYPES = {"task3", "task5", "task8", "task9", "task10"}


@dataclass
class EvalResult:
    total: int = 0
    correct: int = 0
    success_rate: float = 0.0
    per_task_sr: dict[str, float] = field(default_factory=dict)
    query_sr: float = 0.0
    action_sr: float = 0.0
    invalid_action_rate: float = 0.0
    limit_reached_rate: float = 0.0
    avg_steps: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Overall SR: {self.success_rate:.1%} ({self.correct}/{self.total})",
            f"Query SR:   {self.query_sr:.1%}",
            f"Action SR:  {self.action_sr:.1%}",
            f"Invalid:    {self.invalid_action_rate:.1%}",
            f"Limit hit:  {self.limit_reached_rate:.1%}",
            f"Avg steps:  {self.avg_steps:.2f}",
            "Per-task SR:",
        ]
        for task_type in sorted(self.per_task_sr, key=lambda t: int(t.replace("task", ""))):
            lines.append(f"  {task_type}: {self.per_task_sr[task_type]:.1%}")
        return "\n".join(lines)


def _task_type(task_id: str) -> str:
    """Extract task type prefix, e.g. 'task10' from 'task10_3' or 'train_task10_3'."""
    parts = task_id.split("_")
    for part in parts:
        if part.startswith("task"):
            return part
    return parts[0]


def compute_metrics(trajectories: list[Trajectory]) -> EvalResult:
    """Compute evaluation metrics from a list of trajectories."""
    total = len(trajectories)
    if total == 0:
        return EvalResult()

    correct = sum(1 for t in trajectories if t.correct)
    invalid = sum(1 for t in trajectories if t.status == "invalid_action")
    limit_reached = sum(1 for t in trajectories if t.status == "limit_reached")
    total_steps = sum(t.num_steps for t in trajectories)

    # Per-task-type breakdown
    type_correct: dict[str, int] = {}
    type_total: dict[str, int] = {}
    for traj in trajectories:
        tt = _task_type(traj.task_id)
        type_total[tt] = type_total.get(tt, 0) + 1
        if traj.correct:
            type_correct[tt] = type_correct.get(tt, 0) + 1

    per_task_sr: dict[str, float] = {}
    for tt, cnt in type_total.items():
        per_task_sr[tt] = type_correct.get(tt, 0) / cnt

    # Query vs Action SR
    query_correct = sum(type_correct.get(tt, 0) for tt in QUERY_TASK_TYPES if tt in type_total)
    query_total = sum(type_total.get(tt, 0) for tt in QUERY_TASK_TYPES if tt in type_total)
    action_correct = sum(type_correct.get(tt, 0) for tt in ACTION_TASK_TYPES if tt in type_total)
    action_total = sum(type_total.get(tt, 0) for tt in ACTION_TASK_TYPES if tt in type_total)

    return EvalResult(
        total=total,
        correct=correct,
        success_rate=correct / total,
        per_task_sr=per_task_sr,
        query_sr=query_correct / query_total if query_total else 0.0,
        action_sr=action_correct / action_total if action_total else 0.0,
        invalid_action_rate=invalid / total,
        limit_reached_rate=limit_reached / total,
        avg_steps=total_steps / total,
    )
