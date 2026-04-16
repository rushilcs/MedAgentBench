from __future__ import annotations

import json
import logging
from typing import Any

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from rl_training.env.medagent_env import MedAgentEnv
from rl_training.env.reward import RewardConfig, compute_episode_reward
from rl_training.agent.base_policy import BasePolicy
from rl_training.agent.openai_policy import OpenAIPolicy
from rl_training.data.trajectory import Trajectory
from .metrics import EvalResult, compute_metrics

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate a policy on the MedAgentBench benchmark."""

    def __init__(self, env: MedAgentEnv, benchmark_tasks: list[dict[str, Any]]):
        self.env = env
        self.benchmark_tasks = benchmark_tasks

    def _rollout(self, policy: BasePolicy, task: dict[str, Any]) -> Trajectory:
        """Run a single episode and return the trajectory."""
        state = self.env.reset(task)
        while not state.done:
            action = policy.act(state.history)
            result = self.env.step(action)
            state = result.state

        correct = self.env.grade() if state.status == "completed" else False
        traj = Trajectory.from_env_history(
            task=task,
            history=state.history,
            correct=correct,
            status=state.status,
            step_rewards=self.env.step_rewards,
            model_id=getattr(policy, "model_id", ""),
        )
        traj.reward = compute_episode_reward(traj, correct, self.env.reward_config)
        return traj

    def evaluate(self, model_id: str, temperature: float = 0.0) -> EvalResult:
        """Run the model on all benchmark tasks and return metrics."""
        policy = OpenAIPolicy(model_id=model_id, temperature=temperature)
        return self.evaluate_with_policy(policy)

    def evaluate_with_policy(self, policy: BasePolicy) -> EvalResult:
        """Run an arbitrary policy on all benchmark tasks."""
        trajectories: list[Trajectory] = []
        failed = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
        ) as progress:
            eval_task = progress.add_task("Evaluating", total=len(self.benchmark_tasks))
            for task in self.benchmark_tasks:
                try:
                    traj = self._rollout(policy, task)
                    trajectories.append(traj)
                except Exception as exc:
                    logger.error("Rollout failed for %s: %s", task.get("id", "?"), exc)
                    failed += 1
                    trajectories.append(Trajectory.from_env_history(
                        task=task, history=[], correct=False, status="error",
                        model_id=getattr(policy, "model_id", ""),
                    ))
                progress.advance(eval_task)

        if failed:
            logger.warning("%d/%d tasks failed during evaluation", failed, len(self.benchmark_tasks))
        return compute_metrics(trajectories)

    def evaluate_subset(self, model_id: str, task_types: list[int], temperature: float = 0.0) -> EvalResult:
        """Evaluate only on specific task types."""
        prefixes = {f"task{t}" for t in task_types}
        subset = [t for t in self.benchmark_tasks if any(t["id"].startswith(p + "_") for p in prefixes)]
        policy = OpenAIPolicy(model_id=model_id, temperature=temperature)
        trajectories: list[Trajectory] = []
        for task in subset:
            try:
                trajectories.append(self._rollout(policy, task))
            except Exception as exc:
                logger.error("Rollout failed for %s: %s", task.get("id", "?"), exc)
                trajectories.append(Trajectory.from_env_history(
                    task=task, history=[], correct=False, status="error",
                    model_id=model_id,
                ))
        return compute_metrics(trajectories)
