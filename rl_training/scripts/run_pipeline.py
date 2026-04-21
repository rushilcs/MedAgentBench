#!/usr/bin/env python3
"""End-to-end pipeline: Phase A -> Phase B -> final evaluation."""
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
from rl_training.data.task_generator import TaskGenerator
from rl_training.data.trajectory_store import TrajectoryStore
from rl_training.training.openai_finetune import OpenAIFineTuner
from rl_training.training.sft_trainer import SFTTrainer
from rl_training.training.grpo_trainer import GRPOTrainer, GRPOConfig
from rl_training.evaluation.evaluator import Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full RL training pipeline")
    parser.add_argument("--config", default="rl_training/configs/default.yaml")
    parser.add_argument("--output-dir", default="rl_training/outputs/pipeline")
    parser.add_argument("--skip-phase-a", action="store_true", help="Skip Phase A, use --initial-model instead")
    parser.add_argument("--initial-model", default=None, help="Model ID to start Phase B with (skips Phase A)")
    parser.add_argument(
        "--skip-phase-b",
        action="store_true",
        help="Skip Phase B (iterative GRPO); run baseline + Phase A SFT + evals only",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    with open(config["env"]["data_file"]) as f:
        benchmark_tasks = json.load(f)
    existing_mrns = {t["eval_MRN"] for t in benchmark_tasks if "eval_MRN" in t}

    env = MedAgentEnv.from_config(config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate training tasks
    task_gen = TaskGenerator(
        fhir_api_base=config["env"]["fhir_api_base"],
        seed=config.get("training_data", {}).get("seed", 42),
        existing_mrns=existing_mrns,
    )
    training_tasks = task_gen.generate_all(
        count_per_type=config.get("training_data", {}).get("tasks_per_type", 50)
    )
    with open(out_dir / "training_tasks.json", "w") as f:
        json.dump(training_tasks, f, indent=2)
    logger.info("Generated %d training tasks", len(training_tasks))

    fine_tuner = OpenAIFineTuner()
    evaluator = Evaluator(env=env, benchmark_tasks=benchmark_tasks)

    # --- Baseline evaluation ---
    logger.info("=== Baseline evaluation ===")
    base_model = config.get("phase_a", {}).get("base_model", "gpt-4o-mini")
    baseline_result = evaluator.evaluate(base_model)
    logger.info("Baseline:\n%s", baseline_result.summary())
    with open(out_dir / "baseline_eval.json", "w") as f:
        json.dump({"model_id": base_model, "success_rate": baseline_result.success_rate, "per_task_sr": baseline_result.per_task_sr}, f, indent=2)

    # --- Phase A ---
    if args.skip_phase_a and args.initial_model:
        current_model = args.initial_model
        logger.info("Skipping Phase A, using model: %s", current_model)
    else:
        logger.info("=== Phase A: Supervised Fine-Tuning ===")
        sft_store = TrajectoryStore(out_dir / "phase_a" / "expert_trajectories.jsonl")
        sft_trainer = SFTTrainer(env=env, fine_tuner=fine_tuner, store=sft_store, config=config)
        expert_model = config.get("phase_a", {}).get("expert_model", "gpt-4o")
        current_model = sft_trainer.run(training_tasks, expert_model_id=expert_model)

        phase_a_result = evaluator.evaluate(current_model)
        logger.info("Phase A eval:\n%s", phase_a_result.summary())
        with open(out_dir / "phase_a_eval.json", "w") as f:
            json.dump({"model_id": current_model, "success_rate": phase_a_result.success_rate, "per_task_sr": phase_a_result.per_task_sr}, f, indent=2)

    # --- Phase B ---
    grpo_trainer: GRPOTrainer | None = None
    if args.skip_phase_b:
        logger.info("=== Phase B: skipped (--skip-phase-b) ===")
        final_model = current_model
        training_history: list = []
    else:
        logger.info("=== Phase B: GRPO Iterations ===")
        grpo_store = TrajectoryStore(out_dir / "phase_b" / "grpo_trajectories.jsonl")
        grpo_config = GRPOConfig.from_dict(config.get("phase_b", {}))
        grpo_trainer = GRPOTrainer(
            env=env, fine_tuner=fine_tuner, store=grpo_store, evaluator=evaluator, config=grpo_config,
        )
        final_model = grpo_trainer.run(training_tasks, initial_model_id=current_model)
        training_history = grpo_trainer.get_training_history()

    # --- Final evaluation ---
    logger.info("=== Final evaluation ===")
    final_result = evaluator.evaluate(final_model)
    logger.info("Final:\n%s", final_result.summary())

    # --- Save summary ---
    summary = {
        "baseline": {"model_id": base_model, "success_rate": baseline_result.success_rate},
        "final": {"model_id": final_model, "success_rate": final_result.success_rate},
        "improvement": final_result.success_rate - baseline_result.success_rate,
        "training_history": training_history,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Pipeline complete. Results saved to %s", out_dir)


if __name__ == "__main__":
    main()
