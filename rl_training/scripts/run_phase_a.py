#!/usr/bin/env python3
"""Phase A: Supervised fine-tuning on expert trajectories."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

# Ensure project root is on sys.path
_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl_training.env.medagent_env import MedAgentEnv
from rl_training.data.task_generator import TaskGenerator
from rl_training.data.trajectory_store import TrajectoryStore
from rl_training.training.openai_finetune import OpenAIFineTuner
from rl_training.training.sft_trainer import SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase A: SFT on expert trajectories")
    parser.add_argument("--config", default="rl_training/configs/default.yaml", help="Config file path")
    parser.add_argument("--output-dir", default="rl_training/outputs/phase_a", help="Output directory")
    parser.add_argument("--expert-model", default=None, help="Override expert model ID")
    parser.add_argument("--skip-model-collection", action="store_true", help="Skip model-based collection, use only programmatic")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load benchmark data to extract existing MRNs (for train/test split)
    with open(config["env"]["data_file"]) as f:
        benchmark_tasks = json.load(f)
    existing_mrns = {t["eval_MRN"] for t in benchmark_tasks if "eval_MRN" in t}

    # Build environment
    env = MedAgentEnv.from_config(config)

    # Generate training tasks
    seed = config.get("training_data", {}).get("seed", 42)
    count_per_type = config.get("training_data", {}).get("tasks_per_type", 50)
    task_gen = TaskGenerator(
        fhir_api_base=config["env"]["fhir_api_base"],
        seed=seed,
        existing_mrns=existing_mrns,
    )
    training_tasks = task_gen.generate_all(count_per_type=count_per_type)
    logger.info("Generated %d training tasks", len(training_tasks))

    # Save training tasks
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "training_tasks.json", "w") as f:
        json.dump(training_tasks, f, indent=2)

    # Run Phase A
    store = TrajectoryStore(out_dir / "expert_trajectories.jsonl")
    fine_tuner = OpenAIFineTuner()

    expert_model = args.expert_model or config.get("phase_a", {}).get("expert_model", "gpt-4o")
    if args.skip_model_collection:
        expert_model = ""

    trainer = SFTTrainer(env=env, fine_tuner=fine_tuner, store=store, config=config)
    model_id = trainer.run(training_tasks, expert_model_id=expert_model)

    # Save result
    result = {"phase": "A", "model_id": model_id, "num_training_tasks": len(training_tasks)}
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Phase A complete. Model: %s", model_id)


if __name__ == "__main__":
    main()
