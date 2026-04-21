#!/usr/bin/env python3
"""Post-RFT evaluation of the fine-tuned o4-mini on the MedAgentBench benchmark.

Reads ``<run_dir>/finetuned_model_id.txt`` (produced by
``finetune_o4_mini_rft.py``), runs the full 300-task benchmark through the same
multi-turn harness used for the baseline, and writes
``<run_dir>/02_finetuned_o4_mini_benchmark.json``.
"""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Post-RFT eval of fine-tuned o4-mini")
    parser.add_argument("--config", default="rl_training/configs/o4_mini_rft.yaml")
    parser.add_argument("--model", default=None,
                        help="Override the fine-tuned model id (defaults to <run_dir>/finetuned_model_id.txt)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--reasoning-effort", default=None,
                        choices=["low", "medium", "high"])
    args = parser.parse_args()

    require_openai_key()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_dir = Path(args.output_dir or config["output"]["run_dir"])
    id_path = run_dir / "finetuned_model_id.txt"
    if args.model:
        model_id = args.model
    else:
        if not id_path.exists():
            raise SystemExit(
                f"Missing {id_path}. Either pass --model <ft:...> or run "
                "rl_training/scripts/finetune_o4_mini_rft.py first."
            )
        model_id = id_path.read_text().strip()
    if not model_id:
        raise SystemExit("Empty fine-tuned model id.")

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
    payload["stage"] = "post_rft"

    out_path = run_dir / "02_finetuned_o4_mini_benchmark.json"
    write_eval_json(payload, out_path)

    traj_path = run_dir / "02_post_rft_trajectories.jsonl"
    with open(traj_path, "w") as f:
        for traj in trajectories:
            f.write(traj.to_jsonl_line() + "\n")
    logger.info("Saved %d trajectories -> %s", len(trajectories), traj_path)

    print("\n" + result.summary())
    print(f"\n==> {out_path}")


if __name__ == "__main__":
    main()
