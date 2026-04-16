from __future__ import annotations

import json
import logging
from typing import Any

from rl_training.env.medagent_env import MedAgentEnv
from rl_training.agent.openai_policy import OpenAIPolicy
from rl_training.data.trajectory import Trajectory
from rl_training.data.trajectory_store import TrajectoryStore
from .expert_collector import ExpertCollector
from .openai_finetune import OpenAIFineTuner

logger = logging.getLogger(__name__)


class SFTTrainer:
    """Phase A: supervised fine-tuning on expert trajectories."""

    def __init__(
        self,
        env: MedAgentEnv,
        fine_tuner: OpenAIFineTuner,
        store: TrajectoryStore,
        config: dict[str, Any],
    ):
        self.env = env
        self.fine_tuner = fine_tuner
        self.store = store
        self.config = config

    def run(
        self,
        training_tasks: list[dict[str, Any]],
        expert_model_id: str = "gpt-4o",
    ) -> str:
        """Phase A pipeline.

        1. Build programmatic expert trajectories (guaranteed correct).
        2. Optionally collect additional trajectories from a strong model.
        3. Deduplicate and filter to only correct trajectories.
        4. Fine-tune GPT-4o-mini via OpenAI API.
        5. Return the fine-tuned model ID.
        """
        collector = ExpertCollector(env=self.env, store=self.store)
        all_correct: list[Trajectory] = []

        # --- Programmatic trajectories ---
        if self.config.get("phase_a", {}).get("use_programmatic", True):
            logger.info("Building programmatic expert trajectories...")
            programmatic = collector.collect_programmatic(training_tasks)
            logger.info("Programmatic: %d correct trajectories", len(programmatic))
            all_correct.extend(programmatic)

        # --- Model-based trajectories ---
        if expert_model_id:
            logger.info("Collecting expert trajectories from model %s...", expert_model_id)
            policy = OpenAIPolicy(model_id=expert_model_id, temperature=0.0)
            model_trajs = collector.collect(training_tasks, policy, trajectories_per_task=1)
            logger.info("Model-based: %d correct trajectories", len(model_trajs))
            # Only add tasks not already covered by programmatic
            covered_ids = {t.task_id for t in all_correct}
            for t in model_trajs:
                if t.task_id not in covered_ids:
                    all_correct.append(t)

        logger.info("Total expert trajectories: %d", len(all_correct))

        if not all_correct:
            raise RuntimeError("No correct expert trajectories collected. Cannot fine-tune.")

        # --- Fine-tune ---
        base_model = self.config.get("phase_a", {}).get("base_model", "gpt-4o-mini")
        n_epochs = self.config.get("phase_a", {}).get("ft_epochs", 3)
        model_id = self.fine_tuner.run(
            trajectories=all_correct,
            base_model=base_model,
            suffix="medagent-sft",
            n_epochs=n_epochs,
        )
        logger.info("Phase A complete. Fine-tuned model: %s", model_id)
        return model_id
