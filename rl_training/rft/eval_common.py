"""Shared helpers for benchmarking OpenAI models (baseline and post-RFT).

Wraps ``Evaluator`` with lightweight task-level parallelism so reasoning models
(which have high per-request latency) don't take hours on the 300-task bench.

Produces a JSON payload with the exact schema used by
``rl_training/outputs/gpt4o_mini_user_run/01_baseline_gpt4o_mini_ootb.json`` so
pre/post reports line up cleanly.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
    TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)

from rl_training.agent.openai_policy import OpenAIPolicy
from rl_training.data.trajectory import Trajectory
from rl_training.env.medagent_env import MedAgentEnv
from rl_training.evaluation.evaluator import Evaluator
from rl_training.evaluation.metrics import EvalResult, compute_metrics

logger = logging.getLogger(__name__)


def _load_config(path: str | Path) -> dict:
    import yaml  # local import so unit tests don't require it at module load
    with open(path) as f:
        return yaml.safe_load(f)


def load_benchmark_tasks(config: dict) -> list[dict[str, Any]]:
    data_file = config["env"]["data_file"]
    with open(data_file) as f:
        tasks = json.load(f)
    return tasks


def _rollout_one(
    config: dict,
    task: dict,
    policy: OpenAIPolicy,
    worker_local: threading.local,
) -> Trajectory:
    """Per-task rollout using a thread-local env so workers don't collide."""
    env = getattr(worker_local, "env", None)
    if env is None:
        env = MedAgentEnv.from_config(config)
        worker_local.env = env  # type: ignore[attr-defined]
        worker_local.evaluator = Evaluator(env=env, benchmark_tasks=[])  # type: ignore[attr-defined]
    evaluator: Evaluator = worker_local.evaluator
    return evaluator._rollout(policy, task)  # noqa: SLF001


def run_bench_eval(
    *,
    model_id: str,
    config: dict,
    max_parallel: int = 5,
    max_tokens: int = 4096,
    reasoning_effort: str = "medium",
    task_limit: int | None = None,
) -> tuple[EvalResult, list[Trajectory], dict]:
    """Run the full 300-task benchmark and return (metrics, trajectories, payload)."""
    tasks = load_benchmark_tasks(config)
    if task_limit is not None:
        tasks = tasks[: task_limit]
    logger.info("Benchmarking %s on %d tasks (parallel=%d)", model_id, len(tasks), max_parallel)

    policy = OpenAIPolicy(
        model_id=model_id,
        temperature=0.0,
        max_tokens=max_tokens,
        max_parallel=max_parallel,
        reasoning_effort=reasoning_effort,
    )
    worker_local = threading.local()

    trajectories: list[Trajectory | None] = [None] * len(tasks)
    correct_count = 0
    failed = 0

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
        ptask = progress.add_task(
            f"Evaluating {model_id}", total=len(tasks), sr=0.0,
        )
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            future_to_idx = {
                executor.submit(_rollout_one, config, task, policy, worker_local): i
                for i, task in enumerate(tasks)
            }
            done = 0
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                try:
                    traj = future.result()
                    trajectories[i] = traj
                    if traj.correct:
                        correct_count += 1
                except Exception as exc:
                    logger.error("Rollout failed for %s: %s", tasks[i].get("id"), exc)
                    trajectories[i] = Trajectory.from_env_history(
                        task=tasks[i], history=[], correct=False,
                        status="error", model_id=model_id,
                    )
                    failed += 1
                done += 1
                progress.update(ptask, advance=1, sr=correct_count / max(1, done))

    if failed:
        logger.warning("%d/%d tasks failed during evaluation", failed, len(tasks))

    result = compute_metrics([t for t in trajectories if t is not None])
    payload = {
        "model_id": model_id,
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
    return result, [t for t in trajectories if t is not None], payload


def write_eval_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote eval JSON to %s", path)


def ensure_repo_on_path() -> None:
    """Add the repository root to sys.path so ``from rl_training import ...`` works when scripts are run directly."""
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)


def require_openai_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set in the environment.")
