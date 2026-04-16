"""Reward functions for MedAgentBench RL training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl_training.data.trajectory import Trajectory


@dataclass
class RewardConfig:
    correct_bonus: float = 1.0
    incorrect_penalty: float = -0.5
    invalid_action_penalty: float = -1.0
    step_penalty: float = -0.02
    efficiency_bonus: float = 0.1
    max_steps_for_bonus: int = 4


def compute_step_reward(action_kind: str, valid: bool) -> float:
    """Compute reward for a single environment step."""
    if not valid or action_kind == "invalid":
        return -0.5
    if action_kind == "finish":
        return 0.0
    return -0.02  # small cost per intermediate step


def compute_episode_reward(
    trajectory: "Trajectory",
    correct: bool,
    config: RewardConfig | None = None,
) -> float:
    """Compute the total reward for a completed episode."""
    if config is None:
        config = RewardConfig()

    reward = 0.0

    if correct:
        reward += config.correct_bonus
        if trajectory.num_steps <= config.max_steps_for_bonus:
            reward += config.efficiency_bonus
    else:
        reward += config.incorrect_penalty

    if trajectory.status == "invalid_action":
        reward += config.invalid_action_penalty

    reward += config.step_penalty * trajectory.num_steps

    return reward
