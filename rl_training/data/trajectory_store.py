from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

from .trajectory import Trajectory


class TrajectoryStore:
    """JSONL-backed storage for trajectories."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, trajectory: Trajectory) -> None:
        with open(self.path, "a") as f:
            f.write(trajectory.to_jsonl_line() + "\n")

    def save_batch(self, trajectories: list[Trajectory]) -> None:
        with open(self.path, "a") as f:
            for traj in trajectories:
                f.write(traj.to_jsonl_line() + "\n")

    def load_all(self) -> list[Trajectory]:
        if not self.path.exists():
            return []
        trajectories: list[Trajectory] = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    trajectories.append(Trajectory.from_dict(json.loads(line)))
        return trajectories

    def filter(
        self,
        *,
        correct: bool | None = None,
        task_type: str | None = None,
        min_reward: float | None = None,
        predicate: Callable[[Trajectory], bool] | None = None,
    ) -> list[Trajectory]:
        """Load and filter trajectories."""
        all_trajs = self.load_all()
        result: list[Trajectory] = []
        for traj in all_trajs:
            if correct is not None and traj.correct != correct:
                continue
            if task_type is not None and not traj.task_id.startswith(task_type + "_"):
                continue
            if min_reward is not None and traj.reward < min_reward:
                continue
            if predicate is not None and not predicate(traj):
                continue
            result.append(traj)
        return result

    def count(self) -> int:
        if not self.path.exists():
            return 0
        count = 0
        with open(self.path) as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def export_openai_jsonl(self, output_path: str | Path, trajectories: list[Trajectory] | None = None) -> Path:
        """Export trajectories as an OpenAI fine-tuning JSONL file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        trajs = trajectories if trajectories is not None else self.load_all()
        with open(out, "w") as f:
            for traj in trajs:
                f.write(traj.to_openai_jsonl_line() + "\n")
        return out
