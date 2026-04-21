#!/usr/bin/env python3
"""Evaluate a Qwen3 (or any vLLM-served) model on the 300-task benchmark.

Assumes a vLLM OpenAI-compatible server is already running (see
``launch_vllm_server.sh``). This script connects to it via
``VLLMPolicy`` and runs the full benchmark, writing results + clinical
metrics to ``--output-dir``.

Typical usage:

    # in one terminal, start the vLLM server:
    CUDA_VISIBLE_DEVICES=1 bash rl_training/scripts/launch_vllm_server.sh

    # in another terminal, run the eval:
    python rl_training/scripts/run_baseline_eval.py \
        --model Qwen/Qwen3-32B-Instruct \
        --output-dir rl_training/outputs/qwen3_32b_baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl_training.agent.vllm_policy import VLLMPolicy
from rl_training.env.medagent_env import MedAgentEnv
from rl_training.evaluation.clinical_metrics import (
    compute_clinical_metrics,
    save_clinical_metrics,
)
from rl_training.evaluation.evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline eval with vLLM")
    parser.add_argument("--config", default="rl_training/configs/default.yaml",
                        help="Base config (env section is consumed)")
    parser.add_argument("--model", required=True,
                        help="Model id (must match --model on the vLLM server)")
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:8000/v1",
                        help="vLLM OpenAI-compatible endpoint")
    parser.add_argument("--data-file", default=None,
                        help="Override benchmark tasks JSON")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for eval.json + clinical.json")
    parser.add_argument("--task-types", nargs="*", type=int, default=None,
                        help="Evaluate only specific task types (e.g. 4 6 7)")
    parser.add_argument("--max-parallel", type=int, default=8,
                        help="Concurrent rollouts against vLLM")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_file = args.data_file or config["env"]["data_file"]
    with open(data_file) as f:
        benchmark_tasks = json.load(f)
    logger.info("Loaded %d benchmark tasks from %s", len(benchmark_tasks), data_file)

    if args.task_types:
        prefixes = {f"task{t}" for t in args.task_types}
        benchmark_tasks = [
            t for t in benchmark_tasks
            if any(t["id"].startswith(p + "_") for p in prefixes)
        ]
        logger.info("Filtered to %d tasks of types %s",
                    len(benchmark_tasks), args.task_types)

    # Build one shared policy (stateless; VLLMPolicy uses an internal async pool).
    os.environ["VLLM_BASE_URL"] = args.vllm_base_url
    policy = VLLMPolicy(
        model_id=args.model,
        base_url=args.vllm_base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_parallel=args.max_parallel,
    )

    # Build a shared template evaluator just to borrow _rollout; each task gets
    # its own MedAgentEnv instance (env holds per-episode mutable state).
    logger.info("Starting eval: model=%s url=%s parallel=%d (task-level)",
                args.model, args.vllm_base_url, args.max_parallel)

    from rl_training.data.trajectory import Trajectory
    from rl_training.evaluation.metrics import compute_metrics
    from rich.progress import (
        BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
        TextColumn, TimeElapsedColumn, TimeRemainingColumn,
    )
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _rollout_one(task: dict) -> Trajectory:
        env = MedAgentEnv.from_config(config)
        evaluator = Evaluator(env=env, benchmark_tasks=[task])
        return evaluator._rollout(policy, task)  # noqa: SLF001

    trajectories: list[Trajectory] = []
    correct_count = 0
    done_count = 0
    # Progress-JSONL so monitoring scripts can read completion count live.
    prog_path = Path(args.output_dir) / "progress.jsonl"
    prog_path.parent.mkdir(parents=True, exist_ok=True)
    prog_fh = open(prog_path, "w", buffering=1)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("  SR={task.fields[sr]:.1%}"),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
    ) as progress:
        pt = progress.add_task(
            "Evaluating", total=len(benchmark_tasks), sr=0.0,
        )
        with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
            futures = {pool.submit(_rollout_one, t): t for t in benchmark_tasks}
            for fut in as_completed(futures):
                task = futures[fut]
                try:
                    traj = fut.result()
                    trajectories.append(traj)
                    if traj.correct:
                        correct_count += 1
                except Exception as exc:
                    logger.warning("Rollout failed for %s: %s", task.get("id"), exc)
                    trajectories.append(Trajectory.from_env_history(
                        task=task, history=[], correct=False, status="error",
                        model_id=args.model,
                    ))
                done_count += 1
                sr = correct_count / max(1, done_count)
                progress.update(pt, advance=1, sr=sr)
                prog_fh.write(json.dumps({
                    "done": done_count,
                    "total": len(benchmark_tasks),
                    "correct": correct_count,
                    "sr": sr,
                }) + "\n")
    prog_fh.close()

    result = compute_metrics(trajectories)
    print("\n" + result.summary())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_payload = {
        "model_id": args.model,
        "vllm_base_url": args.vllm_base_url,
        "total": result.total,
        "correct": result.correct,
        "success_rate": result.success_rate,
        "per_task_sr": result.per_task_sr,
        "query_sr": result.query_sr,
        "action_sr": result.action_sr,
        "invalid_action_rate": result.invalid_action_rate,
        "limit_reached_rate": result.limit_reached_rate,
        "avg_steps": result.avg_steps,
    }
    with open(out_dir / "eval.json", "w") as f:
        json.dump(eval_payload, f, indent=2)
    logger.info("Saved eval summary to %s", out_dir / "eval.json")

    clinical = compute_clinical_metrics(trajectories)
    save_clinical_metrics(clinical, str(out_dir / "clinical.json"))
    print("\n" + clinical.summary())
    logger.info("Saved clinical metrics to %s", out_dir / "clinical.json")

    trajectories_path = out_dir / "trajectories.jsonl"
    with open(trajectories_path, "w") as f:
        for traj in trajectories:
            f.write(traj.to_jsonl_line() + "\n")
    logger.info("Saved %d trajectories to %s",
                len(trajectories), trajectories_path)


if __name__ == "__main__":
    main()
