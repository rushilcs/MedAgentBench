from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from rl_training.env.medagent_env import MedAgentEnv
from rl_training.env.reward import compute_episode_reward
from rl_training.agent.base_policy import BasePolicy
from rl_training.agent.openai_policy import OpenAIPolicy
from rl_training.data.trajectory import Trajectory
from rl_training.data.trajectory_store import TrajectoryStore
from rl_training.evaluation.evaluator import Evaluator
from rl_training.evaluation.metrics import EvalResult
from .openai_finetune import OpenAIFineTuner

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    trajectories_per_task: int = 5
    temperature: float = 0.7
    num_iterations: int = 5
    selection_method: str = "above_mean"  # "above_mean" | "top_k" | "top_percent"
    top_k: int = 2
    top_percent: float = 0.5
    min_trajectories: int = 50
    ft_epochs: int = 3
    base_model: str = "gpt-4o-mini"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GRPOConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class IterationLog:
    iteration: int
    model_id: str
    total_rollouts: int
    selected_count: int
    correct_count: int
    eval_result: EvalResult | None = None


class GRPOTrainer:
    """Phase B: iterative rejection-sampling GRPO."""

    def __init__(
        self,
        env: MedAgentEnv,
        fine_tuner: OpenAIFineTuner,
        store: TrajectoryStore,
        evaluator: Evaluator,
        config: GRPOConfig,
    ):
        self.env = env
        self.fine_tuner = fine_tuner
        self.store = store
        self.evaluator = evaluator
        self.config = config
        self.logs: list[IterationLog] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        training_tasks: list[dict[str, Any]],
        initial_model_id: str,
    ) -> str:
        """Run the full GRPO loop and return the best model ID."""
        current_model = initial_model_id

        for iteration in range(self.config.num_iterations):
            logger.info("=== GRPO iteration %d / %d  (model: %s) ===", iteration + 1, self.config.num_iterations, current_model)

            # 1. Rollout
            policy = OpenAIPolicy(model_id=current_model, temperature=self.config.temperature)
            all_trajectories = self._rollout_all(policy, training_tasks)
            correct_count = sum(1 for t in all_trajectories if t.correct)
            logger.info("Rollout: %d trajectories, %d correct (%.1f%%)", len(all_trajectories), correct_count, 100 * correct_count / max(len(all_trajectories), 1))

            # 2. Score
            for traj in all_trajectories:
                traj.reward = compute_episode_reward(traj, traj.correct, self.env.reward_config)

            # 3. GRPO selection
            selected = self._grpo_select(all_trajectories)
            # Only keep correct trajectories for fine-tuning
            selected = [t for t in selected if t.correct]
            logger.info("Selected %d trajectories for fine-tuning", len(selected))

            # 4. Fine-tune (if we have enough data)
            if len(selected) >= self.config.min_trajectories:
                new_model = self.fine_tuner.run(
                    trajectories=selected,
                    base_model=self.config.base_model,
                    suffix=f"medagent-grpo-i{iteration}",
                    n_epochs=self.config.ft_epochs,
                )
                current_model = new_model
                logger.info("Fine-tuned new model: %s", new_model)
            else:
                logger.warning("Only %d selected trajectories (< %d min). Skipping fine-tune.", len(selected), self.config.min_trajectories)

            # 5. Evaluate
            eval_result = self.evaluator.evaluate(current_model, temperature=0.0)
            logger.info("Evaluation:\n%s", eval_result.summary())

            # 6. Log
            log_entry = IterationLog(
                iteration=iteration,
                model_id=current_model,
                total_rollouts=len(all_trajectories),
                selected_count=len(selected),
                correct_count=correct_count,
                eval_result=eval_result,
            )
            self.logs.append(log_entry)

            # Save trajectories from this iteration
            self.store.save_batch(all_trajectories)

        return current_model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rollout_all(
        self, policy: BasePolicy, tasks: list[dict[str, Any]]
    ) -> list[Trajectory]:
        """Rollout K trajectories for each task."""
        all_trajs: list[Trajectory] = []
        total = len(tasks) * self.config.trajectories_per_task
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn()) as prog:
            ptask = prog.add_task("Rolling out", total=total)
            for task in tasks:
                for _ in range(self.config.trajectories_per_task):
                    traj = self._rollout_single(policy, task)
                    all_trajs.append(traj)
                    prog.advance(ptask)
        return all_trajs

    def _rollout_single(self, policy: BasePolicy, task: dict[str, Any]) -> Trajectory:
        """Run one episode."""
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

    def _grpo_select(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        """GRPO-style selection: keep trajectories with above-mean reward per task."""
        by_task: dict[str, list[Trajectory]] = defaultdict(list)
        for traj in trajectories:
            by_task[traj.task_id].append(traj)

        selected: list[Trajectory] = []
        for task_id, group in by_task.items():
            if not group:
                continue
            mean_reward = sum(t.reward for t in group) / len(group)

            if self.config.selection_method == "above_mean":
                selected.extend(t for t in group if t.reward > mean_reward)
            elif self.config.selection_method == "top_k":
                sorted_group = sorted(group, key=lambda t: t.reward, reverse=True)
                selected.extend(sorted_group[: self.config.top_k])
            elif self.config.selection_method == "top_percent":
                sorted_group = sorted(group, key=lambda t: t.reward, reverse=True)
                n = max(1, int(len(sorted_group) * self.config.top_percent))
                selected.extend(sorted_group[:n])

        return selected

    def get_training_history(self) -> list[dict[str, Any]]:
        """Return a summary of all iterations for logging / plotting."""
        history = []
        for log in self.logs:
            entry: dict[str, Any] = {
                "iteration": log.iteration,
                "model_id": log.model_id,
                "total_rollouts": log.total_rollouts,
                "selected_count": log.selected_count,
                "correct_count": log.correct_count,
            }
            if log.eval_result:
                entry["success_rate"] = log.eval_result.success_rate
                entry["per_task_sr"] = log.eval_result.per_task_sr
                entry["action_sr"] = log.eval_result.action_sr
                entry["query_sr"] = log.eval_result.query_sr
            history.append(entry)
        return history
