#!/usr/bin/env python3
"""Resume the pipeline from after expert trajectory collection.

Loads already-collected trajectories, fine-tunes gpt-4o-2024-08-06,
then runs Phase B (GRPO iterations) and final evaluation.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import yaml

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl_training.env.medagent_env import MedAgentEnv
from rl_training.data.trajectory_store import TrajectoryStore
from rl_training.training.openai_finetune import OpenAIFineTuner
from rl_training.training.grpo_trainer import GRPOTrainer, GRPOConfig
from rl_training.evaluation.evaluator import Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def main() -> None:
    out_dir = Path("rl_training/outputs/gpt4o_pipeline")
    config_path = "rl_training/configs/default.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    with open(config["env"]["data_file"]) as f:
        benchmark_tasks = json.load(f)

    env = MedAgentEnv.from_config(config)
    fine_tuner = OpenAIFineTuner()
    evaluator = Evaluator(env=env, benchmark_tasks=benchmark_tasks)

    # Load saved trajectories
    store = TrajectoryStore(out_dir / "phase_a" / "expert_trajectories.jsonl")
    all_trajs = store.load_all()
    correct_trajs = [t for t in all_trajs if t.correct]
    logger.info("Loaded %d trajectories (%d correct)", len(all_trajs), len(correct_trajs))

    if not correct_trajs:
        raise RuntimeError("No correct trajectories found")

    # Phase A: Fine-tune
    base_model = config.get("phase_a", {}).get("base_model", "gpt-4o-2024-08-06")
    n_epochs = config.get("phase_a", {}).get("ft_epochs", 3)
    logger.info("=== Phase A: Fine-tuning %s on %d trajectories ===", base_model, len(correct_trajs))

    current_model = fine_tuner.run(
        trajectories=correct_trajs,
        base_model=base_model,
        suffix="medagent-sft",
        n_epochs=n_epochs,
    )
    logger.info("Phase A model: %s", current_model)

    # Evaluate Phase A
    phase_a_result = evaluator.evaluate(current_model)
    logger.info("Phase A eval:\n%s", phase_a_result.summary())
    with open(out_dir / "phase_a_eval.json", "w") as f:
        json.dump({
            "model_id": current_model,
            "success_rate": phase_a_result.success_rate,
            "per_task_sr": phase_a_result.per_task_sr,
        }, f, indent=2)

    # Phase B: GRPO iterations
    logger.info("=== Phase B: GRPO Iterations ===")
    with open(out_dir / "training_tasks.json") as f:
        training_tasks = json.load(f)

    grpo_store = TrajectoryStore(out_dir / "phase_b" / "grpo_trajectories.jsonl")
    grpo_config = GRPOConfig.from_dict(config.get("phase_b", {}))
    grpo_trainer = GRPOTrainer(
        env=env, fine_tuner=fine_tuner, store=grpo_store, evaluator=evaluator, config=grpo_config,
    )
    final_model = grpo_trainer.run(training_tasks, initial_model_id=current_model)

    # Final evaluation
    logger.info("=== Final evaluation ===")
    final_result = evaluator.evaluate(final_model)
    logger.info("Final:\n%s", final_result.summary())

    # Save summary
    baseline = json.load(open(out_dir / "baseline_eval.json"))
    summary = {
        "baseline": baseline,
        "phase_a": {"model_id": current_model, "success_rate": phase_a_result.success_rate},
        "final": {"model_id": final_model, "success_rate": final_result.success_rate},
        "improvement": final_result.success_rate - baseline["success_rate"],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Pipeline complete. Results saved to %s", out_dir)


if __name__ == "__main__":
    main()
