#!/usr/bin/env python3
"""Phase B: GRPO-style iterative rejection-sampling fine-tuning."""
from __future__ import annotations

import argparse
import json
import logging
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
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase B: GRPO iterations")
    parser.add_argument("--config", default="rl_training/configs/default.yaml")
    parser.add_argument("--initial-model", required=True, help="Model ID from Phase A (or base model)")
    parser.add_argument("--training-tasks", required=True, help="Path to training_tasks.json from Phase A")
    parser.add_argument("--output-dir", default="rl_training/outputs/phase_b")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    with open(args.training_tasks) as f:
        training_tasks = json.load(f)

    with open(config["env"]["data_file"]) as f:
        benchmark_tasks = json.load(f)

    env = MedAgentEnv.from_config(config)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    store = TrajectoryStore(out_dir / "grpo_trajectories.jsonl")
    fine_tuner = OpenAIFineTuner()
    evaluator = Evaluator(env=env, benchmark_tasks=benchmark_tasks)

    grpo_config = GRPOConfig.from_dict(config.get("phase_b", {}))
    trainer = GRPOTrainer(
        env=env,
        fine_tuner=fine_tuner,
        store=store,
        evaluator=evaluator,
        config=grpo_config,
    )

    final_model = trainer.run(training_tasks, initial_model_id=args.initial_model)

    # Save results
    result = {
        "phase": "B",
        "final_model_id": final_model,
        "initial_model_id": args.initial_model,
        "training_history": trainer.get_training_history(),
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Phase B complete. Final model: %s", final_model)


if __name__ == "__main__":
    main()
