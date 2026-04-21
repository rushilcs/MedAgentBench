#!/usr/bin/env python3
"""Baseline evaluation of ``o4-mini-2025-04-16`` on the MedAgentBench 300-task benchmark.

Writes the eval summary to ``<run_dir>/01_baseline_o4_mini_ootb.json`` with the
same schema as ``rl_training/outputs/gpt4o_mini_user_run/01_baseline_gpt4o_mini_ootb.json``.
"""

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

from rl_training.rft.eval_common import (
    require_openai_key,
    run_bench_eval,
    write_eval_json,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline o4-mini eval on MedAgentBench")
    parser.add_argument("--config", default="rl_training/configs/o4_mini_rft.yaml")
    parser.add_argument("--model", default=None, help="Override base model id")
    parser.add_argument("--output-dir", default=None,
                        help="Override run dir (defaults to config.output.run_dir)")
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--task-limit", type=int, default=None,
                        help="Smoke-test only: cap number of benchmark tasks")
    parser.add_argument("--reasoning-effort", default=None,
                        choices=["low", "medium", "high"])
    args = parser.parse_args()

    require_openai_key()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_id = args.model or config["rft"]["base_model"]
    run_dir = Path(args.output_dir or config["output"]["run_dir"])
    max_parallel = args.max_parallel or config.get("evaluation", {}).get("max_parallel", 5)
    max_tokens = config.get("evaluation", {}).get("max_tokens", 4096)
    reasoning_effort = args.reasoning_effort or config["rft"].get("reasoning_effort", "medium")

    result, trajectories, payload = run_bench_eval(
        model_id=model_id,
        config=config,
        max_parallel=max_parallel,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        task_limit=args.task_limit,
    )
    payload["reasoning_effort"] = reasoning_effort
    payload["stage"] = "baseline"

    out_path = run_dir / "01_baseline_o4_mini_ootb.json"
    write_eval_json(payload, out_path)

    traj_path = run_dir / "01_baseline_trajectories.jsonl"
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    with open(traj_path, "w") as f:
        for traj in trajectories:
            f.write(traj.to_jsonl_line() + "\n")
    logger.info("Saved %d trajectories -> %s", len(trajectories), traj_path)

    print("\n" + result.summary())
    print(f"\n==> {out_path}")


if __name__ == "__main__":
    main()
