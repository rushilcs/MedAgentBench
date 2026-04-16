#!/usr/bin/env python3
"""Standalone evaluation of a model on the MedAgentBench benchmark."""
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
from rl_training.evaluation.evaluator import Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a model on MedAgentBench")
    parser.add_argument("--config", default="rl_training/configs/default.yaml")
    parser.add_argument("--model", required=True, help="Model ID to evaluate")
    parser.add_argument("--output", default=None, help="Path to save results JSON")
    parser.add_argument("--task-types", nargs="*", type=int, default=None, help="Evaluate only specific task types")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    with open(config["env"]["data_file"]) as f:
        benchmark_tasks = json.load(f)

    env = MedAgentEnv.from_config(config)
    evaluator = Evaluator(env=env, benchmark_tasks=benchmark_tasks)

    if args.task_types:
        result = evaluator.evaluate_subset(args.model, args.task_types)
    else:
        result = evaluator.evaluate(args.model)

    print("\n" + result.summary())

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({
                "model_id": args.model,
                "total": result.total,
                "correct": result.correct,
                "success_rate": result.success_rate,
                "per_task_sr": result.per_task_sr,
                "query_sr": result.query_sr,
                "action_sr": result.action_sr,
                "invalid_action_rate": result.invalid_action_rate,
                "avg_steps": result.avg_steps,
            }, f, indent=2)
        logger.info("Saved results to %s", out)


if __name__ == "__main__":
    main()
