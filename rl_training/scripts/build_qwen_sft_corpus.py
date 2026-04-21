#!/usr/bin/env python3
"""Build the Qwen3-32B SFT corpus (run2).

Replaces ``generate_qwen_sft_expert_trajectories.py`` with the same pattern
used by the gpt-4o-mini phase-A pipeline (`run_phase_a.py` ->
`SFTTrainer.run`). Concretely:

* Generate ``count_per_type * 10`` training tasks via ``TaskGenerator``.
* Build **programmatic** trajectories for all 10 task types via
  ``ExpertCollector.collect_programmatic`` (guaranteed correct).
* Run **model-based** rollouts with gpt-4o (or any chat model) for
  additional diversity. Rollouts are parallelized across tasks via
  ``ThreadPoolExecutor`` (each worker has its own ``MedAgentEnv``).
* Filter to correct, deduplicate by message-content hash, export to
  ``qwen_sft_openai.jsonl`` (the format ``sft_qwen3_32b.py`` consumes).

The script is resumable: existing programmatic and model trajectories in
``expert_trajectories.jsonl`` are kept; we only re-run rollouts for tasks
missing from the store. Rerun with ``--no-skip-existing`` to force a
full regeneration.

Cost / time estimate (with default ``--model-trajectories-per-task 2``):
* 500 programmatic trajectories: ~1 min, ~$0
* 1000 gpt-4o rollouts (~5 turns each, ~3K input + ~150 output tokens):
  ~10M input + ~1M output tokens => ~$50-65, ~30-60 min wall time at
  ``--max-parallel 8``
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from threading import Lock

import yaml

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl_training.agent.openai_policy import OpenAIPolicy
from rl_training.data.task_generator import TaskGenerator
from rl_training.data.trajectory import Trajectory
from rl_training.data.trajectory_store import TrajectoryStore
from rl_training.env.medagent_env import MedAgentEnv
from rl_training.env.reward import compute_episode_reward
from rl_training.training.expert_collector import ExpertCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_or_generate_training_tasks(
    config: dict,
    explicit_path: str | None,
    output_dir: Path,
    count_per_type: int,
    seed: int,
) -> list[dict]:
    if explicit_path and Path(explicit_path).exists():
        with open(explicit_path) as f:
            tasks = json.load(f)
        logger.info("Loaded %d training tasks from %s", len(tasks), explicit_path)
        return tasks

    out_path = output_dir / "training_tasks.json"
    if out_path.exists():
        with open(out_path) as f:
            tasks = json.load(f)
        logger.info("Reusing existing %s (%d tasks)", out_path, len(tasks))
        return tasks

    logger.info(
        "Generating fresh training tasks: count_per_type=%d, seed=%d (FHIR=%s)",
        count_per_type, seed, config["env"]["fhir_api_base"],
    )
    with open(config["env"]["data_file"]) as f:
        benchmark_tasks = json.load(f)
    existing_mrns = {t["eval_MRN"] for t in benchmark_tasks if "eval_MRN" in t}
    task_gen = TaskGenerator(
        fhir_api_base=config["env"]["fhir_api_base"],
        seed=seed,
        existing_mrns=existing_mrns,
    )
    tasks = task_gen.generate_all(count_per_type=count_per_type)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tasks, indent=2))
    logger.info("Generated + saved %d training tasks to %s", len(tasks), out_path)
    return tasks


def _hash_messages(messages: list[dict]) -> str:
    """Stable content hash of a chat-completion message list (for dedup)."""
    payload = json.dumps(
        [{"role": m["role"], "content": m["content"]} for m in messages],
        sort_keys=False,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _build_env(config: dict) -> MedAgentEnv:
    return MedAgentEnv.from_config(config)


# ---------------------------------------------------------------------------
# Parallel model-based rollout
# ---------------------------------------------------------------------------

def _rollout_one(
    env: MedAgentEnv,
    task: dict,
    policy: OpenAIPolicy,
) -> Trajectory:
    state = env.reset(task)
    while not state.done:
        action = policy.act(state.history)
        result = env.step(action)
        state = result.state
    correct = env.grade() if state.status == "completed" else False
    traj = Trajectory.from_env_history(
        task=task,
        history=state.history,
        correct=correct,
        status=state.status,
        step_rewards=env.step_rewards,
        model_id=getattr(policy, "model_id", ""),
    )
    traj.reward = compute_episode_reward(traj, correct, env.reward_config)
    return traj


def _collect_model_parallel(
    config: dict,
    tasks: list[dict],
    policy: OpenAIPolicy,
    trajectories_per_task: int,
    max_workers: int,
    store: TrajectoryStore,
    save_lock: Lock,
) -> tuple[int, int]:
    """Run gpt-4o rollouts in parallel with one MedAgentEnv per worker."""
    # Build a dedicated env per worker to avoid shared mutable state.
    envs: list[MedAgentEnv] = [_build_env(config) for _ in range(max_workers)]
    env_pool: list[MedAgentEnv] = list(envs)
    pool_lock = Lock()

    def _acquire_env() -> MedAgentEnv:
        with pool_lock:
            return env_pool.pop()

    def _release_env(e: MedAgentEnv) -> None:
        with pool_lock:
            env_pool.append(e)

    def _job(task: dict) -> Trajectory:
        env = _acquire_env()
        try:
            return _rollout_one(env, task, policy)
        finally:
            _release_env(env)

    # Build the rollout list: trajectories_per_task copies of each task.
    items: list[dict] = []
    for t in tasks:
        for _ in range(trajectories_per_task):
            items.append(t)

    total = len(items)
    correct = 0
    attempted = 0
    start_t = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_job, it): idx for idx, it in enumerate(items)}
        for fut in as_completed(futures):
            attempted += 1
            try:
                traj = fut.result()
            except Exception as exc:
                logger.warning("Rollout failed: %s", exc)
                continue
            if traj.correct:
                correct += 1
            with save_lock:
                store.save(traj)
            if attempted % 25 == 0 or attempted == total:
                elapsed = time.time() - start_t
                rate = attempted / max(elapsed, 1e-3)
                eta = (total - attempted) / max(rate, 1e-3)
                logger.info(
                    "Model rollouts: %d/%d (correct=%d, %.1f/s, eta=%.0fs)",
                    attempted, total, correct, rate, eta,
                )
    return attempted, correct


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build Qwen3-32B SFT corpus (run2)")
    parser.add_argument("--config", default="rl_training/configs/default.yaml",
                        help="Base config file (env/training_data sections used)")
    parser.add_argument("--training-tasks", default=None,
                        help="Optional explicit path to training_tasks.json")
    parser.add_argument("--output-dir", default="rl_training/outputs/qwen_pipeline_v2/phase_a",
                        help="Where to write expert_trajectories.jsonl + qwen_sft_openai.jsonl")
    parser.add_argument("--count-per-type", type=int, default=50,
                        help="Number of training tasks per task type (10 types -> total)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expert-model", default="gpt-4o",
                        help="OpenAI chat model id used as the expert policy")
    parser.add_argument("--model-trajectories-per-task", type=int, default=2,
                        help="Number of independent gpt-4o rollouts per task")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for the expert policy")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-parallel", type=int, default=8,
                        help="Concurrent rollouts (also bounds OpenAI requests)")
    parser.add_argument("--skip-programmatic", action="store_true",
                        help="Skip the programmatic-trajectory pass")
    parser.add_argument("--skip-model", action="store_true",
                        help="Skip the model-based-rollout pass")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true",
                        default=True,
                        help="Skip rollouts for task_ids already in the store")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--limit-tasks", type=int, default=None,
                        help="Dev flag: only process the first N training tasks")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY") and not args.skip_model:
        logger.error("OPENAI_API_KEY is not set; either set it or pass --skip-model")
        sys.exit(2)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "expert_trajectories.jsonl"
    sft_path = out_dir / "qwen_sft_openai.jsonl"
    summary_path = out_dir / "generation_summary.json"

    tasks = _load_or_generate_training_tasks(
        config, args.training_tasks, out_dir,
        count_per_type=args.count_per_type,
        seed=args.seed,
    )
    if args.limit_tasks:
        tasks = tasks[: args.limit_tasks]
        logger.info("--limit-tasks: truncated to %d tasks", len(tasks))

    store = TrajectoryStore(traj_path)
    save_lock = Lock()
    existing = store.load_all()
    existing_ids = {t.task_id for t in existing}
    existing_correct = sum(1 for t in existing if t.correct)
    logger.info(
        "Existing store: %d trajectories (%d correct), covering %d task ids",
        len(existing), existing_correct, len(existing_ids),
    )

    # ----- Programmatic pass -----
    programmatic_count = 0
    if not args.skip_programmatic:
        env = _build_env(config)
        collector = ExpertCollector(env=env, store=store)
        # Avoid re-running programmatic for tasks already covered by a
        # programmatic trajectory (model_id=='programmatic').
        programmatic_done = {
            t.task_id for t in existing
            if t.model_id == "programmatic" and t.correct
        }
        prog_targets = [t for t in tasks if t["id"] not in programmatic_done] \
            if args.skip_existing else list(tasks)
        logger.info(
            "Programmatic pass: %d tasks to process (%d already done)",
            len(prog_targets), len(programmatic_done),
        )
        if prog_targets:
            built = collector.collect_programmatic(prog_targets)
            programmatic_count = sum(1 for t in built if t.correct)
            logger.info("Programmatic pass: built %d new correct trajectories", programmatic_count)

    # ----- Model-based pass -----
    attempted = 0
    new_correct = 0
    if not args.skip_model:
        # Skip model rollouts for tasks that already have a model-based correct
        # trajectory (so we naturally stop at trajectories_per_task).
        if args.skip_existing:
            model_done_counts: dict[str, int] = {}
            for t in store.load_all():
                if t.model_id != "programmatic" and t.correct:
                    model_done_counts[t.task_id] = model_done_counts.get(t.task_id, 0) + 1
            model_targets = [
                t for t in tasks
                if model_done_counts.get(t["id"], 0) < args.model_trajectories_per_task
            ]
        else:
            model_targets = list(tasks)
        logger.info(
            "Model pass: %d tasks * %d trajs = up to %d rollouts",
            len(model_targets), args.model_trajectories_per_task,
            len(model_targets) * args.model_trajectories_per_task,
        )
        if model_targets:
            policy = OpenAIPolicy(
                model_id=args.expert_model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_parallel=args.max_parallel,
            )
            attempted, new_correct = _collect_model_parallel(
                config, model_targets, policy,
                trajectories_per_task=args.model_trajectories_per_task,
                max_workers=args.max_parallel,
                store=store,
                save_lock=save_lock,
            )
            logger.info(
                "Model pass: attempted=%d correct=%d", attempted, new_correct,
            )

    # ----- Filter, dedupe, export -----
    all_correct = store.filter(correct=True)
    logger.info("Total correct trajectories in store: %d", len(all_correct))

    seen: set[str] = set()
    deduped: list[Trajectory] = []
    for tr in all_correct:
        msgs = tr.to_openai_messages()
        h = _hash_messages(msgs)
        if h in seen:
            continue
        seen.add(h)
        deduped.append(tr)
    logger.info("After content-hash dedup: %d trajectories", len(deduped))

    # Per-task-type breakdown
    by_type: dict[str, int] = {}
    for tr in deduped:
        # task ids look like 'train_task5_3'
        parts = tr.task_id.split("_")
        ttype = next((p for p in parts if p.startswith("task")), "unknown")
        by_type[ttype] = by_type.get(ttype, 0) + 1
    logger.info("Per-task-type breakdown: %s", json.dumps(by_type, sort_keys=True))

    store.export_openai_jsonl(sft_path, deduped)
    logger.info("Wrote %d deduped trajectories to %s", len(deduped), sft_path)

    summary = {
        "expert_model": args.expert_model,
        "model_trajectories_per_task": args.model_trajectories_per_task,
        "temperature": args.temperature,
        "count_per_type": args.count_per_type,
        "tasks_total": len(tasks),
        "existing_trajectories_at_start": len(existing),
        "programmatic_new_correct": programmatic_count,
        "model_attempted": attempted,
        "model_new_correct": new_correct,
        "store_total_correct": len(all_correct),
        "deduped_total": len(deduped),
        "per_task_type_after_dedup": by_type,
        "expert_trajectories_path": str(traj_path),
        "qwen_sft_openai_path": str(sft_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote generation summary to %s", summary_path)

    if len(deduped) == 0:
        logger.error("No correct trajectories produced; aborting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
